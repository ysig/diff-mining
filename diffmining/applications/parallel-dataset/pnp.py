import os
import sys
import argparse
import yaml
import random
import joblib
from os.path import join
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm, trange
import PIL
from PIL import Image
import joblib
from typing import Optional

# Numerical Libraries
import math
import numpy as np
import pandas as pd
import cv2

# Neural Networks and Deep Learning Libraries
import torch
import torch.nn as nn
import torchvision.transforms as T
import xformers
import xformers.ops
from torch.utils.data import Dataset
from torch.nn.functional import interpolate, cosine_similarity
from torchvision.transforms import CenterCrop
from torchvision.transforms.functional import to_tensor
from einops import rearrange
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionPipeline
from diffusers.utils import USE_PEFT_BACKEND
from diffusers.models.attention_processor import Attention


SAVE_STEPS = 1000
STEPS = 999
SEED = 42

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx_start', type=int, required=False, default=0)
    parser.add_argument('--k_start', type=int, required=False, default=0)
    parser.add_argument('--k_end', type=int, required=False, default=1000)
    parser.add_argument('--batch_size', type=int, required=False, default=10)
    parser.add_argument('--cache', type=str, required=False, default='.cache/')
    parser.add_argument('--save_dir', type=str, required=False, default='dataset/parallel')
    parser.add_argument('--model_path', type=str, required=False, default='models/export')
    parser.add_argument('--base_path', type=str, required=False, default='dataset/base')

    args = parser.parse_args()
    CACHE = args.cache
    MODEL = args.model_path
    save_dir = args.save_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def register_time(model, t):
    res_dict = {
        0: [0, 1, 2],
        1: [0, 1, 2],
        2: [0, 1, 2],
        3: [0, 1, 2]
    }  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            conv_module = model.unet.up_blocks[res].resnets[block]
            setattr(conv_module, 't', t)

    for res in range(len(model.unet.down_blocks)):
        for block in range(len(model.unet.down_blocks[res].resnets)):
            conv_module = model.unet.down_blocks[res].resnets[block]
            setattr(conv_module, 't', t)

    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)

    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)

    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', t)

    n_res = model.unet.mid_block.resnets
    for i in range(len(n_res)):
        module = model.unet.mid_block.resnets[i]
        setattr(module, 't', t)

def get_timesteps(scheduler, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    return timesteps, num_inference_steps - t_start

class Preprocess(nn.Module):
    def __init__(self, device, hf_key=None):
        super().__init__()

        self.device = device
        self.use_depth = False

        # Create model
        self.vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained('geolocal/StreetCLIP')
        self.text_encoder = CLIPTextModel.from_pretrained('geolocal/StreetCLIP').to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet").to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(MODEL, subfolder="scheduler")
        self.inversion_func = self.ddim_inversion

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def load_img(self, image_path):
        image_pil = T.Resize(512)(Image.open(image_path).convert("RGB"))
        image = T.ToTensor()(image_pil).unsqueeze(0).to(device)
        return image

    @torch.no_grad()
    def encode_imgs(self, imgs):
        imgs = imgs.unsqueeze(0).to(self.device)
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.mean * 0.18215
        return latents

    @torch.no_grad()
    def ddim_inversion(self, cond, latent, save_latents=True, timesteps_to_save=None):
        timesteps = reversed(self.scheduler.timesteps)
        latents_ = {}
        for i, t in enumerate(timesteps):
            cond_batch = cond.repeat(latent.shape[0], 1, 1)

            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[timesteps[i - 1]]
                if i > 0 else self.scheduler.final_alpha_cumprod
            )

            mu = alpha_prod_t ** 0.5
            mu_prev = alpha_prod_t_prev ** 0.5
            sigma = (1 - alpha_prod_t) ** 0.5
            sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

            eps = self.unet(latent, t, encoder_hidden_states=cond_batch).sample

            pred_x0 = (latent - sigma_prev * eps) / mu_prev
            latent = mu * pred_x0 + sigma * eps
            latents_[t.item()] = latent.clone().cpu()

        return latent, latents_

    @torch.no_grad()
    def ddim_sample(self, x, cond, save_latents=False, timesteps_to_save=None):
        timesteps = self.scheduler.timesteps
        for i, t in enumerate(timesteps):
                cond_batch = cond.repeat(x.shape[0], 1, 1)
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i + 1]]
                    if i < len(timesteps) - 1
                    else self.scheduler.final_alpha_cumprod
                )
                mu = alpha_prod_t ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(x, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (x - sigma * eps) / mu
                x = mu_prev * pred_x0 + sigma_prev * eps

        return x

    @torch.no_grad()
    def extract_latents(self, image, num_steps, timesteps_to_save, inversion_prompt='', extract_reverse=False):
        self.scheduler.set_timesteps(num_steps)

        cond = self.get_text_embeds(inversion_prompt, "")[1].unsqueeze(0)
        latent = self.encode_imgs(image)

        inverted_x, latents = self.inversion_func(cond, latent, timesteps_to_save=timesteps_to_save)
        latent_reconstruction = self.ddim_sample(inverted_x, cond, timesteps_to_save=timesteps_to_save)
        rgb_reconstruction = self.decode_latents(latent_reconstruction)

        return rgb_reconstruction, inverted_x, latents  # , latent_reconstruction

def extract_latents(image, inversion_prompt):
    toy_scheduler = DDIMScheduler.from_pretrained(MODEL, subfolder="scheduler")
    toy_scheduler.set_timesteps(SAVE_STEPS)
    timesteps_to_save, num_inference_steps = get_timesteps(
        toy_scheduler, num_inference_steps=SAVE_STEPS, strength=1.0, device=device)

    seed_everything(SEED)
    extraction_path_prefix = "_forward" #"_reverse" if opt.extract_reverse else

    model = Preprocess(device)
    recon_image, latents, latents_ = model.extract_latents(
        image, num_steps=STEPS, timesteps_to_save=timesteps_to_save,
        inversion_prompt=inversion_prompt, extract_reverse=True
    )

    return latents, latents_, T.ToPILImage()(recon_image[0])

def process_image(img, crop):
    img = Image.open(img)
    # w, h = img.size
    # if w < h:
    #     w, h = 512, int(h / w * 512)
    # else:
    #     w, h = int(w / h * 512), 512

    # img = img.resize((w, h), Image.Resampling.LANCZOS)
    # # img = CenterCrop(512)(img)
    # if crop is not None:
    #   img = img.crop(crop)
    pil = img
    img = to_tensor(img)*2-1.0
    return img, pil

def get_latents(image_path, format_text, crop=None, region=False):
    country = os.path.split(os.path.split(image_path)[0])[1]
    image, pil = process_image(image_path, crop=crop)
    prompt = format_text(country)
    latents, latents_, recon_image = extract_latents(image, prompt)
    return {'latents': (latents, latents_), 'rec': recon_image, 'country': country, 'prompt': prompt, 'pil': pil}

def extract(x, target, block, res, time):
    image, features, selfattn = infer(x['latents'][0], x['latents'][1], target, block, res, time)
    return {'features': features, 'selfattn': selfattn, 'image': image}

def get_seed(path, format_text=lambda x: '', crop=None, region=False):
    os.makedirs(CACHE, exist_ok=True)
    pre_head = '_'.join(os.path.split(path)[-1].split('_')[1:])
    tag = ('' if crop is None else f'{crop[0]}_{crop[1]}_')
    tagp = ('' if not len(format_text('<country>')) else format_text('<country>') + '_')
    x_file = os.path.join(CACHE, f'{tag}{tagp}{region}{pre_head}.pkl')
    if not os.path.isfile(x_file):
        x = get_latents(path, format_text, crop=crop, region=region)
        # joblib.dump(x, x_file)
    else:
        x = joblib.load(x_file)
    return x

def register_conv_control_efficient(model, injection_schedule, f):
    def conv_forward(self):
        def forward(
            input_tensor: torch.FloatTensor,
            temb: torch.FloatTensor,
            scale: float = 1.0,
        ) -> torch.FloatTensor:
            hidden_states = input_tensor

            if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
                hidden_states = self.norm1(hidden_states, temb)
            else:
                hidden_states = self.norm1(hidden_states)

            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = (
                    self.upsample(input_tensor, scale=scale)
                    if isinstance(self.upsample, Upsample2D)
                    else self.upsample(input_tensor)
                )
                hidden_states = (
                    self.upsample(hidden_states, scale=scale)
                    if isinstance(self.upsample, Upsample2D)
                    else self.upsample(hidden_states)
                )
            elif self.downsample is not None:
                input_tensor = (
                    self.downsample(input_tensor, scale=scale)
                    if isinstance(self.downsample, Downsample2D)
                    else self.downsample(input_tensor)
                )
                hidden_states = (
                    self.downsample(hidden_states, scale=scale)
                    if isinstance(self.downsample, Downsample2D)
                    else self.downsample(hidden_states)
                )

            hidden_states = self.conv1(hidden_states, scale) if not USE_PEFT_BACKEND else self.conv1(hidden_states)

            if self.time_emb_proj is not None:
                if not self.skip_time_act:
                    temb = self.nonlinearity(temb)
                temb = (
                    self.time_emb_proj(temb, scale)[:, :, None, None]
                    if not USE_PEFT_BACKEND
                    else self.time_emb_proj(temb)[:, :, None, None]
                )

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
                hidden_states = self.norm2(hidden_states, temb)
            else:
                hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states, scale) if not USE_PEFT_BACKEND else self.conv2(hidden_states)
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(hidden_states.shape[0] // 3)
                # inject unconditional
                hidden_states[source_batch_size:2 * source_batch_size] = hidden_states[:source_batch_size]
                # inject conditional
                hidden_states[2 * source_batch_size:] = hidden_states[:source_batch_size]

            if self.conv_shortcut is not None:
                input_tensor = (
                    self.conv_shortcut(input_tensor, scale) if not USE_PEFT_BACKEND else self.conv_shortcut(input_tensor)
                )

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor
        return forward

    n_res = model.unet.mid_block.resnets
    for i in range(len(n_res)):
        if f(-1, i):
            conv_module = model.unet.mid_block.resnets[i]
            conv_module.forward = conv_forward(conv_module)
            setattr(conv_module, 'injection_schedule', injection_schedule)

    for res in range(len(model.unet.up_blocks)):
        for block in range(len(model.unet.up_blocks[res].resnets)):
            if f(res, block):
                conv_module = model.unet.up_blocks[res].resnets[block]
                conv_module.forward = conv_forward(conv_module)
                setattr(conv_module, 'injection_schedule', injection_schedule)

def register_attention_control_efficient(model, injection_schedule, g):
    def sa_forward(attn):
        def forward(
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
        ) -> torch.FloatTensor:
            residual = hidden_states

            args = () if USE_PEFT_BACKEND else (scale,)

            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, key_tokens, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )

            attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
            if attention_mask is not None:
                # expand our mask's singleton query_tokens dimension:
                #   [batch*heads,            1, key_tokens] ->
                #   [batch*heads, query_tokens, key_tokens]
                # so that it can be added as a bias onto the attention scores that xformers computes:
                #   [batch*heads, query_tokens, key_tokens]
                # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
                _, query_tokens, _ = hidden_states.shape
                attention_mask = attention_mask.expand(-1, query_tokens, -1)

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            is_cross = encoder_hidden_states is not None
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            query = attn.to_q(hidden_states, *args)
            key = attn.to_k(encoder_hidden_states, *args)
            if not is_cross and attn.injection_schedule is not None and (attn.t in attn.injection_schedule or attn.t == 1000):
                source_batch_size = int(query.shape[0] // 3)
                # inject unconditional
                query[source_batch_size:2*source_batch_size] = query[:source_batch_size]
                key[source_batch_size:2*source_batch_size] = key[:source_batch_size]

                # inject conditional
                query[2*source_batch_size:] = query[:source_batch_size]
                key[2*source_batch_size:] = key[:source_batch_size]

            query = attn.head_to_batch_dim(query).contiguous()
            key = attn.head_to_batch_dim(key).contiguous()

            value = attn.to_v(encoder_hidden_states, *args)
            value = attn.head_to_batch_dim(value).contiguous()

            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask, op=attn.processor.attention_op, scale=attn.scale
            )
            hidden_states = hidden_states.to(query.dtype)
            hidden_states = attn.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states, *args)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states

        return forward

    n_res = model.unet.mid_block.attentions
    for block in range(len(n_res)):
        if g(-1, block):
            module = model.unet.mid_block.attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)

    for res in range(len(model.unet.up_blocks)):
        if hasattr(model.unet.up_blocks[res], 'attentions'):
            for block in range(len(model.unet.up_blocks[res].attentions)):
                if g(res, block):
                    module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
                    module.forward = sa_forward(module)
                    setattr(module, 'injection_schedule', injection_schedule)


class PNP(object):
    def __init__(self, latent, lat_t, uncond, pnp_attn_t=0.5, pnp_f_t=0.8, base_prompt=None):
        self.config = {
            'seed': 1,
            'guidance_scale': 7.5,
            'n_timesteps': 50,
            'pnp_attn_t': pnp_attn_t,
            'pnp_f_t': pnp_f_t,
        }
        self.device = device

        # Create SD models
        pipe = StableDiffusionPipeline.from_pretrained(MODEL,
            tokenizer=CLIPTokenizer.from_pretrained('geolocal/StreetCLIP'),
            text_encoder=CLIPTextModel.from_pretrained('geolocal/StreetCLIP'),
        ).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(MODEL, subfolder="scheduler")
        self.scheduler.set_timesteps(self.config["n_timesteps"], device=self.device)

        # load image
        self.eps, self.lat = latent, lat_t
        self.pnp_guidance_embeds_ = self.get_text_embeds([uncond])
        self.pnp_guidance_embeds_base = self.get_text_embeds([base_prompt])

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        return self.text_encoder(text_input.input_ids.to(self.device))[0]

    def load(self, prompt):
        B = len(prompt)
        if any(len(p) == 0 for p in prompt):
            self.pnp_guidance_embeds = []
            for p in prompt:
                if len(p):
                    self.pnp_guidance_embeds.append(self.pnp_guidance_embeds_)
                else:
                    self.pnp_guidance_embeds.append(self.pnp_guidance_embeds_base)
            self.text_embeds = torch.cat([torch.cat(self.pnp_guidance_embeds, dim=0), self.get_text_embeds(prompt)], dim=0)
            self.pnp_guidance_embeds = self.pnp_guidance_embeds_.expand(B, -1, -1)
        else:
            self.pnp_guidance_embeds = self.pnp_guidance_embeds_.expand(B, -1, -1)
            self.text_embeds = torch.cat([self.pnp_guidance_embeds, self.get_text_embeds(prompt)], dim=0)

    @torch.no_grad()
    def decode_latent(self, latent):
        latent = 1 / 0.18215 * latent
        img = self.vae.decode(latent).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        return img

    @torch.no_grad()
    def denoise_step(self, x, t):
        # register the time step and features in pnp injection modules
        source_latents = self.lat[t.item()].to(device)

        latent_model_input = torch.cat([source_latents]*self.pnp_guidance_embeds.shape[0] + ([x] * 2))
        register_time(self, t.item())

        # compute text embeddings
        text_embed_input = torch.cat([self.pnp_guidance_embeds, self.text_embeds], dim=0)

        # apply the denoising network
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']

        # perform guidance
        _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        noise_pred = noise_pred_uncond + self.config["guidance_scale"] * (noise_pred_cond - noise_pred_uncond)

        # compute the denoising step with the reference model
        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']
        return denoised_latent

    def init_pnp(self, conv_injection_t, qk_injection_t, time, f, g):
        self.qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_attention_control_efficient(self, self.qk_injection_timesteps, g=g)
        register_conv_control_efficient(self, self.conv_injection_timesteps, f=f)

    def run_pnp(self, time, f, g):
        pnp_f_t = int(self.config["n_timesteps"] * self.config["pnp_f_t"])
        pnp_attn_t = int(self.config["n_timesteps"] * self.config["pnp_attn_t"])
        self.init_pnp(pnp_f_t, pnp_attn_t, time=time, f=f, g=g)
        return self.sample_loop(self.eps)

    def sample_loop(self, x):
        x = torch.cat([x] * (self.text_embeds.shape[0]//2), dim=0)
        for t in self.scheduler.timesteps:
            x = self.denoise_step(x, t)

        return self.decode_latent(x)


class Generator(object):
    def __init__(self, path, preformat_text=lambda x: '', precrop=False, crop=None, pnp_attn_t=0.5, pnp_f_t=0.8, uncond_ignore=False, region=False):
        self.crop = crop
        self.precrop = precrop
        self.x = get_seed(path, preformat_text, crop=(crop if precrop else None), region=region)
        self.pre_head = '_'.join(os.path.split(path)[-1].split('_')[1:])
        self.pnp = PNP(self.x['latents'][0], self.x['latents'][1], (self.x['prompt'] if not uncond_ignore else ''), pnp_attn_t=pnp_attn_t, pnp_f_t=pnp_f_t, base_prompt=self.x['prompt'])
        self.transform = (self.pilify_crop if self.crop is not None and not self.precrop else self.pilify)
        self.uncond_ignore = uncond_ignore

    def pilify(self, x):
        return Image.fromarray((x*255).astype(np.uint8))

    def pilify_crop(self, x):
        return self.pilify(x).crop(self.crop)

    def pre(self, targets):
        self.save = None
        self.pnp.load(list(map(format_text, targets)))

    def export(self, x):
        x = x.permute(0, 2, 3, 1).cpu().numpy()        
        x = list(map(self.transform, (x[i] for i in range(x.shape[0]))))
        return x

    def generate(self, targets, time=501, format_text='{}'.format, f=lambda *args: False, g=lambda *args: False):
        self.pre(targets)
        xp = self.pnp.run_pnp(time=time, f=f, g=g)
        return self.export(xp)


def plotum(dir_path, module, batch_size=1, format_text='{}'.format, preformat_text=lambda x: '', precrop=False, countries=[]):
  rbf = {1: [1]}
  rbg = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
  f = lambda res, block: block in rbf.get(res, [])
  g = lambda res, block: block in rbg.get(res, [])
  # take the parent dir of dir_path
  parent_dir = os.path.split(dir_path)[0]

  module.x['rec'].save(join(dir_path, 'inverted--' + module.x['country'] + '_' + module.pre_head))
  module.x['pil'].save(join(dir_path, 'gt--' + module.x['country'] + '_' + module.pre_head))

  for countries_batch in list(batchify(countries, batch_size)):
    for c, image in zip(countries_batch, module.generate(countries_batch, f=f, g=g, format_text=format_text)):
        if c == module.x['country']:
            image.save(join(dir_path, f'projected--{c}_{module.pre_head}'))
        else:
            image.save(join(dir_path, f'{c}_{module.pre_head}'))

def batchify(x, b):
  N = len(x)
  for i in range(0, N, b):
      yield x[i:min(i + b, N)]

def format_text(x):
    return f'{x}'

def preformat_text(x):
    return f'{x}'

if __name__ == '__main__':
    idx_start = args.idx_start
    k_start = args.k_start
    k_end = args.k_end
    batch_size = args.batch_size
    COUNTRIES = ['United States', 'Japan', 'France', 'Italy', 'United Kingdom', 'Brazil', 'Russia', 'Thailand', 'Nigeria', 'India']
    for country in COUNTRIES[idx_start:idx_start+1]:
        path = os.path.join(args.base_path, country)
        countries_rest = COUNTRIES
        for f in tqdm(os.listdir(path)[k_start:k_end]):
            image_path = os.path.join(path, f)
            dir_path = join(save_dir, country)
            os.makedirs(dir_path, exist_ok=True)

            pre_head = '_'.join(os.path.split(image_path)[-1].split('_')[1:])
            pathos = [
                join(dir_path, 'inverted--' + country + '_' + pre_head),
                join(dir_path, 'gt--' + country + '_' + pre_head)
            ]
            for c_ in countries_rest:
                if c_ == country:
                    pathos.append(join(dir_path, f'projected--{c_}_{pre_head}'))
                else:
                    pathos.append(join(dir_path, f'{c_}_{pre_head}'))

            if not all(os.path.isfile(p) for p in pathos):
                G = Generator(image_path, preformat_text, precrop=False, uncond_ignore=True, region=False)
                plotum(dir_path, G, batch_size=batch_size, format_text=format_text, preformat_text=preformat_text, precrop=False, countries=countries_rest)
                del G
                torch.cuda.empty_cache()
