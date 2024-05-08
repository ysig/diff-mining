import math
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import gc
import json
import random
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import pandas as pd
import numpy as np

from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop, CenterCrop, Normalize
from torchvision.transforms.functional import to_tensor
from tqdm.auto import tqdm
from os.path import join
from einops import rearrange

from transformers import AutoProcessor,AutoModel
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.training_utils import EMAModel
from diffusers.utils.torch_utils import randn_tensor
from diffmining.finetuning.base import BaseTrainer, concatenate_pil_images_width

NEGATIVE_PROMPT = 'A car'

def tokenize(tokenizer, prompts):
    return tokenizer(prompts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids

def get_decade(year):
    return str((int(year) // 10) * 10)

class CarDB(Dataset):
    def __init__(self, data_path):
        self.data = []
        self.metadata = json.load(open(join(data_path, 'train.json')))
        for image in os.listdir(join(data_path, 'train')):
            self.data.append((join(data_path, 'train', image), get_decade(self.metadata[image]['year'])))

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14-336")

    def __len__(self) -> int:
        return len(self.data)

    def process_image(self, img):
        # rescale to 512 height
        if img.width > img.height:
            img = img.resize((int(img.width * (256 / img.height)), 256), Image.LANCZOS)
        else:
            img = img.resize((256, int(img.height * (256 / img.width))), Image.LANCZOS)

        img = to_tensor(img)
        img = RandomCrop((256, 256))(img)
        img = 2*img - 1.0
        return img

    def __getitem__(self, i):
        path, time = self.data[i]
        img = self.process_image(Image.open(path).convert('RGB'))

        prompt = str(NEGATIVE_PROMPT)
        i = np.random.choice(2, p=[0.05, 0.95])

        if i >= 1:
            prompt = prompt + ' from the ' + time + 's.'
        else:
            prompt = prompt + '.'

        tokenized = tokenize(self.tokenizer, [prompt])
        return dict(image=img, prompt=prompt, description=prompt, tokenized=tokenized)


def init_dataloader(data_path, random_subset=None, shuffle=True, batch_size=8, num_workers=10):
    data = CarDB(data_path)

    if random_subset is not None:
        seed, num_samples = random_subset
        random.seed(seed)
        random.seed(42)
        assert num_samples <= len(data)
        ids = random.sample(range(len(data)), num_samples)
        data = torch.utils.data.Subset(data, ids)

    def collate_fn(examples):
        image = torch.stack([example["image"] for example in examples])
        image = image.to(memory_format=torch.contiguous_format).float()
        prompt = [example["prompt"] for example in examples]
        tokenized = torch.stack([example["tokenized"].squeeze(0) for example in examples])
        return {"image": image, 'prompt': prompt, 'description': prompt, 'tokenized': tokenized}

    # DataLoaders creation:
    dataloader = torch.utils.data.DataLoader(data, shuffle=shuffle, collate_fn=collate_fn, batch_size=batch_size, num_workers=num_workers)

    return {'data': data, 'dataloader': dataloader} | ({'ids': ids} if random_subset is not None else {})

class Trainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.countries = ['1880', '1940', '1980', '2000', '2010']
        self.init_model()
        self.customized_saving_accelerate()

    def init_model(self):
        # Load scheduler, tokenizer and models.
        args = self.args
        self.unet = UNet2DConditionModel.from_pretrained(args.base_name_or_path, subfolder="unet")
        self.resolution = 32
 
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.base_name_or_path, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(args.base_name_or_path, subfolder='vae')
        self.clip = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14-336")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14-336")

        # Freeze vae and text_encoder
        self.vae.requires_grad_(False)
        self.clip.requires_grad_(False)

        # Create EMA for the unet.
        if self.args.use_ema:
            ema_unet = UNet2DConditionModel.from_pretrained(self.args.base_name_or_path, subfolder="unet")
            self.ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    def gradient_checkpointing(self):
        if self.args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

    def init_dataloader(self):
        x = init_dataloader(self.args.data_path, batch_size=self.args.train_batch_size, num_workers=self.args.dataloader_num_workers)
        self.train_dataset, self.train_dataloader = x['data'], x['dataloader']

    def training_init(self):
        # Prepare everything with our `accelerator`.
        self.init_dataloader()
        self.gradient_checkpointing()

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Move text_encode and vae to gpu and cast to weight_dtype
        self.clip = self.clip.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae = self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.init_lora()
        self.init_xformers()
        self.init_optimizer()
        self.init_scheduler()

        if self.args.use_ema:
            self.ema_unet.to(self.accelerator.device, dtype=self.weight_dtype)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        self.num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if self.overrode_max_train_steps:
            self.args.max_train_steps = self.args.num_train_epochs * self.num_update_steps_per_epoch
            self.args.max_train_steps = self.args.num_train_epochs * self.num_update_steps_per_epoch

        if self.args.logging_steps is None:
            self.args.logging_steps = self.num_update_steps_per_epoch // 2
            self.logger.info(f'Logging every {self.args.logging_steps}')

        if self.args.checkpointing_steps is None:
            self.args.checkpointing_steps = self.num_update_steps_per_epoch // 2
            self.logger.info(f'Checkpointing every {self.args.checkpointing_steps}')

        # Afterwards we recalculate our number of training epochs
        self.args.num_train_epochs = math.ceil(self.args.max_train_steps / self.num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("sd-v15-cars", config=vars(self.args))

        self.clip, self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(self.clip, self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler)

        # Train!
        self.total_batch_size = self.args.train_batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps
        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.train_dataset)}")
        self.logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.args.train_batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        self.global_step, self.first_epoch = 0, 0

    def resume_training(self):
        # Potentially load in the weights and states from a previous save
        if self.args.resume_from_checkpoint:
            if self.args.resume_from_checkpoint != "latest":
                path = os.path.basename(self.args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(self.args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(f"Checkpoint '{self.args.resume_from_checkpoint}' does not exist. Starting a new training run.")
                self.args.resume_from_checkpoint = None
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(os.path.join(self.args.output_dir, path))
                self.global_step = int(path.split("-")[1])

                resume_global_step = self.global_step * self.args.gradient_accumulation_steps
                self.first_epoch = self.global_step // self.num_update_steps_per_epoch
                self.resume_step = resume_global_step % (self.num_update_steps_per_epoch * self.args.gradient_accumulation_steps)

    def to_pipeline(self, ddim=False):
        unet = self.accelerator.unwrap_model(self.unet)
        if self.args.use_ema:
            self.ema_unet.copy_to(unet.parameters())

        scheduler = self.noise_scheduler
        if ddim:
            scheduler = DDIMScheduler.from_config(scheduler.config)

        vae = self.accelerator.unwrap_model(self.vae)
        clip = self.accelerator.unwrap_model(self.clip)
        pipeline = StableDiffusionPipeline(vae=vae, text_encoder=clip, tokenizer=self.tokenizer, unet=unet, scheduler=scheduler, feature_extractor=None, safety_checker=None)
        return pipeline

    @torch.no_grad()
    def sample(self, countries, num_samples=5, steps=50, eta='all', seed=42, ddim=True, guidance_scale=7.5):
        pipeline = self.to_pipeline(ddim=ddim)
        if self.args.xformers:
            pipeline.unet.enable_xformers_memory_efficient_attention()
        pipeline.unet.eval()

        latents = None
        if seed is not None:
            torch.manual_seed(seed)
            dim = self.resolution * pipeline.vae_scale_factor
            latents_shape = (num_samples, pipeline.unet.in_channels, dim // 8, dim // 8)
            latents = randn_tensor(latents_shape, device=self.accelerator.device, dtype=self.weight_dtype)

        logs = {}
        for country in countries:
            prompt, negative_prompt = [NEGATIVE_PROMPT + ' at the ' + country + 's.']*num_samples, [NEGATIVE_PROMPT]*num_samples
            with torch.autocast('cuda'):
                logs[f'{country}'] = pipeline(num_inference_steps=steps, eta=0.0, guidance_scale=guidance_scale, latents=latents, prompt=prompt, negative_prompt=negative_prompt, verbose=False)[0]

        return logs

    def step(self, batch):
        with self.accelerator.accumulate(self.unet):
            # Convert images to latent space
            latents = self.vae.encode(batch["image"].to(self.weight_dtype)).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = self.clip(batch["tokenized"])[0]

            # Get the target for loss depending on the prediction type
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
            self.loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = self.accelerator.gather(self.loss.repeat(self.args.train_batch_size)).mean()
            self.train_loss += avg_loss.item() / self.args.gradient_accumulation_steps

            # Backpropagate
            self.accelerator.backward(self.loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.unet.parameters(), self.args.max_grad_norm)

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

    def checkpointing(self):
        # Checks if the accelerator has performed an optimization step behind the scenes
        if self.accelerator.sync_gradients:
            if self.args.use_ema:
                self.ema_unet.step(self.unet.parameters())
            self.progress_bar.update(1)
            self.global_step += 1
            self.accelerator.log({"train_loss": self.train_loss}, step=self.global_step)
            self.train_loss = 0.0

            if self.global_step % self.args.checkpointing_steps == 0:
                if self.accelerator.is_main_process:
                    save_path = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
                    self.accelerator.save_state(save_path)
                    self.logger.info(f"Saved state to {save_path}")

        if self.global_step % self.args.logging_steps == 0:
            if self.accelerator.is_main_process:
                self.save_logs(self.sample(self.countries))
                self.logger.info(f"Saved logs")

    def save_logs(self, logs):
        plot_dir = os.path.join(self.args.output_dir, f"plots", f"{self.global_step}")
        os.makedirs(plot_dir, exist_ok=True)
        for k, v in logs.items():
            concatenate_pil_images_width(v).save(f'{plot_dir}/{k}.png')
        del logs

    def train(self):
        self.training_init()
        self.resume_training()
        args = self.args
        if args.export_only:
            return self.end_training()

        if self.accelerator.is_main_process:
            self.logger.info(f"Computing initial logs")
            self.save_logs(self.sample(self.countries))

        # Only show the progress bar once on each machine.
        self.progress_bar = tqdm(range(self.global_step, args.max_train_steps), disable=not self.accelerator.is_local_main_process)
        self.progress_bar.set_description("Steps")
        for epoch in range(self.first_epoch, args.num_train_epochs):
            self.unet.train()
            self.train_loss = 0.0
            for step, batch in enumerate(self.train_dataloader):
                # Skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == self.first_epoch and step < self.resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        self.progress_bar.update(1)
                    continue

                self.step(batch)
                self.checkpointing()

                self.progress_bar.set_postfix(step_loss=self.loss.detach().item(), lr=self.lr_scheduler.get_last_lr()[0])
                if self.global_step >= args.max_train_steps:
                    break

            if self.global_step >= args.max_train_steps:
                break

        if self.accelerator.is_main_process:
            self.logger.info(f"Computing initial logs")
            self.save_logs(self.sample(self.countries))

        self.end_training()


if __name__ == "__main__":
    Trainer().train()
