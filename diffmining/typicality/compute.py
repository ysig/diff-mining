import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import sys
import json
import PIL
import numpy as np
import torch
import argparse

from collections import defaultdict
from tqdm import tqdm
from os.path import join

from torch.nn import functional as F
from transformers import CLIPTokenizer, CLIPTextModel
from torchvision.transforms.functional import to_tensor
from diffusers import StableDiffusionPipeline


def get_decade(year):
    return str((int(year) // 10) * 10)

class CategoryFeatures(object):
  def __init__(self, clip, tokenizer, device, which):
    super().__init__()
    self.clip = clip
    self.tokenizer = tokenizer
    self.device = device
    self.which = which

  @torch.no_grad()
  def tokenize(self, prompts):
      return self.tokenizer(prompts, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids

  @torch.no_grad()
  def embed(self, categories):
    if self.which == 'faces':
      txt = [(f"Portrait at the {c}'s." if len(c) else "Portrait.") for c in categories]
    elif self.which == 'cars':
      txt = [(f"A car at the {c}'s." if len(c) else "A car.") for c in categories]
    else:
      txt = [(f"{c}" if len(c) else "") for c in categories]

    tokens = self.tokenize(txt)
    return self.clip(tokens.to(self.device))[0].float()

  def __getitem__(self, x):
    return self.embed(x)

class SD(object):
  def __init__(self, which, model_path, categories, device, xformers):
    self.which = which
    self.device = device
    self.clip_name = ('geolocal/StreetCLIP' if which == 'geo' else 'openai/clip-vit-large-patch14-336')

    self.model = StableDiffusionPipeline.from_pretrained(
      model_path,
      tokenizer=CLIPTokenizer.from_pretrained(self.clip_name, torch_dtype=torch.float16),
      text_encoder=CLIPTextModel.from_pretrained(self.clip_name, torch_dtype=torch.float16),
      torch_dtype=torch.float16,
    ).to(self.device)
    if xformers:
      self.model.enable_xformers_memory_efficient_attention()
    self.unet, self.vae, self.clip, self.clip_tokenizer, self.scheduler = self.model.unet.eval(), self.model.vae.eval(), self.model.text_encoder.eval(), self.model.tokenizer, self.model.scheduler

    self.country_features = CategoryFeatures(self.clip, self.clip_tokenizer, self.device, self.which)
    self.categories = sorted(categories)
    apply_categories = [""] + self.categories
    cf = self.country_features[apply_categories]
    self.country_embeds = {c: cf[i] for i, c in enumerate(apply_categories)}

  def copy_model(self):
    return StableDiffusionPipeline.from_pretrained(
      unet=self.unet,
      vae=self.vae,
      text_encoder=self.clip,
      tokenizer=self.clip_tokenizer,
      scheduler=self.scheduler,
      torch_dtype=torch.float16,
    ).to(self.device)

  @torch.autocast('cuda')
  def encode_vae(self, x):
    return self.vae.encode(x.to(self.device)).latent_dist.sample() * self.vae.config.scaling_factor

  @torch.no_grad()
  def compute_loss(self, x, noise, timesteps, c):
    noise = noise.expand(c.size(0), -1, -1, -1)
    with torch.autocast('cuda', dtype=torch.float16):
      noisy_latents = self.scheduler.add_noise(x.expand(c.size(0), -1, -1, -1), noise, timesteps.expand(c.size(0)))
      noise_pred = self.model.unet(noisy_latents, timesteps.expand(c.size(0)), c.to(self.device)).sample
      loss = F.mse_loss(noise_pred.float(), noise, reduction="none")
    return loss


class D(object):
  def __init__(self, sd, typicality_path, which, seed=42, N=100, t_min=0.0, t_max=1.0):
    self.typicality_path = typicality_path
    self.sd = sd
    self.seed = seed
    self.N = N
    self.which = which
    self.t_min = t_min
    self.t_max = t_max

  @torch.no_grad()
  def noising(self, x):
    noise = torch.randn_like(x)
    timesteps = torch.randint(
      int(self.t_min*self.sd.scheduler.num_train_timesteps), 
      int(self.t_max*self.sd.scheduler.num_train_timesteps), (1,),
      device=self.sd.device
    )
    timesteps = timesteps.long()
    return noise, timesteps
  
  def load_image(self, x):
    x = x.convert('RGB')
    with torch.autocast('cuda'):
      x = to_tensor(x)
      x = x * 2 - 1
      x = x.unsqueeze(0)
    return x

  @torch.no_grad()
  def compute_losses(self, img, country_embeds, B=100):
    with torch.inference_mode():
      x = self.sd.encode_vae(self.load_image(img))

      torch.manual_seed(self.seed)
      noises, timesteps = zip(*[self.noising(x) for _ in range(self.N)])
      noises, timesteps = torch.cat(noises, dim=0), torch.cat(timesteps, dim=0)

      losses_grid = []
      n_countries = country_embeds.size(0)
      for i in range(0, noises.shape[0], B):
        n_batch, t_batch = noises[i:i+B].to(self.sd.device), timesteps[i:i+B].to(self.sd.device)
        batch_size = n_batch.size(0)

        # stretch n_batch and t_batch to the number of countries
        n_batch = torch.cat([n_batch]*n_countries, dim=0)
        t_batch = torch.cat([t_batch]*n_countries, dim=0)
        loss_grid = self.sd.compute_loss(x, n_batch, t_batch, torch.cat([country_embeds[c].unsqueeze(0).expand(batch_size, -1, -1) for c in range(n_countries)], dim=0))

        # chunk so that the second dimension is the number of countries
        loss_grid = torch.stack(torch.split(loss_grid, [batch_size]*n_countries, dim=0), dim=1)
        losses_grid.append(loss_grid.cpu())
        
      losses_grid = torch.cat(losses_grid, dim=0)

    return losses_grid.to(dtype=torch.float16)

  def get_path(self, path):
    return join(self.typicality_path, os.path.split(path)[1].replace('.jpg', '.npy').replace('.png', '.npy'))

  def rescale(self, img):
    if self.which == 'cars':
      w, h = img.size
      if w > h:
        w = int(w * 256 / h)
        h = 256
      else:
        h = int(h * 256 / w)
        w = 256
      img = img.resize((w, h), PIL.Image.LANCZOS)
    return img

  def compute(self, country, path):
      img = PIL.Image.open(path)
      seed = os.path.split(path)[1]
      img = self.rescale(img)
      path = join(self.typicality_path, seed.replace('.jpg', '.npy').replace('.png', '.npy'))
      cfs = [self.sd.country_embeds[""]]
      country_embeds = torch.stack([self.sd.country_embeds[country]] + cfs, dim=0)
      os.makedirs(os.path.dirname(path), exist_ok=True)
      losses = self.compute_losses(img, country_embeds)
      path = self.get_path(path)
      np.save(open(path, 'wb'), losses.numpy())

  def __call__(self, path):
    try:
      return np.load(self.get_path(path))
    except ValueError as ve:
      raise ve

  def exists(self, path):
    path = self.get_path(path)
    return os.path.isfile(path)

def get_country(path):
  country = os.path.split(path)[-1].split('__')[0]
  if '--' in country:
    country = country.split('--')[1]
  return country

class Typicality(object):
  def __init__(self, which, model_path, dataset_path, typicality_path, t_min=0.0, t_max=1.0, xformers=True):
    self.which = which
    self.load_paths = (self.load_paths_geo if which == 'geo' else self.load_paths_ftt if which == 'ftt' else self.load_paths_cars)
    self.load_paths(dataset_path)
    self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if model_path is not None:
      self.sd = SD(self.which, model_path, self.categories(), self.device, xformers=xformers)
    else:
      self.sd = None
    self.D = {category: D(self.sd, join(typicality_path, category), t_min=t_min, t_max=t_max, which=which) for category in self.categories()}

  def load_paths_geo(self, dataset_path):
    self.parent = {}
    self.country_path = defaultdict(list)
    for country_parent in os.listdir(dataset_path):
      seed_base, seeds = {}, defaultdict(list)
      output_dir = join(dataset_path, country_parent)
      for seed in os.listdir(output_dir):
        sid = '__'.join(seed.replace('.jpg', '').split('__')[1:])
        country = seed.split('__')[0]
        if country.startswith('gt--'):
          country = country.replace('gt--', '')
          self.country_path[country].append((join(output_dir, seed), True))
          seed_base[sid] = join(output_dir, seed)
        elif '--' not in country:
          self.country_path[country].append((join(output_dir, seed), False))
          seeds[sid].append(join(output_dir, seed))
      self.parent[country_parent] = {'base': seed_base, 'neighbors': seeds}

    self.parallel = defaultdict(list)
    for country, d in tqdm(self.parent.items()):
      for k, v in d['base'].items():
        data = [(v, country)] + [(n, os.path.split(n)[1].split('_')[0]) for n in d['neighbors'][k]]
        self.parallel[country].append(data)

  def load_paths_ftt(self, dataset_path):
    self.parent = {}
    self.times = defaultdict(list)
    for t in os.listdir(dataset_path):
      for path in os.listdir(join(dataset_path, t)):
        self.times[t].append(join(dataset_path, t, path))

  def load_paths_cars(self, dataset_path):
    self.parent = {}
    self.times = defaultdict(list)
    self.metadata = json.load(open(dataset_path + '.json', 'r'))
    path = os.path.split(dataset_path)[0]
    for image in os.listdir(join(dataset_path)):
        self.times[get_decade(self.metadata[image]['year'])].append(join(dataset_path, image))

  def categories(self):
    if self.which == 'geo':
      return self.parent.keys()
    elif self.which == 'ftt':
      return sorted(self.times.keys())
    else:
      return sorted(self.times.keys())

  def compute_submission(self, path):
    with open(path, 'r') as f:
      lines = f.readlines()
    
    for line in tqdm(lines, desc='Executing submission'):
      path, country = line.strip().split(',')
      self.D[country].compute(country, path)

  def get_seeds_(self, c):
    if self.which in {'ftt', 'cars'}:
      return [path for path in self.times[c]]
    elif self.which == 'geo':
      return [path[0] for path in self.country_path[c] if path[1]]

  def make_submission(self, target_path, submission_path, seed=42, sub_split=32):
    full = {c: [] for c in self.categories()}
    state = {c: 0 for c in self.categories()}
    for c in self.categories():
      for path in self.get_seeds_(c):
        if self.D[c].exists(path):
          state[c] += 1
        else:
          full[c].append(path)

    subs = []
    while any(map(len, full.values())):
      category = min(state, key=state.get)

      try:
        path = full[category].pop(0)
      except IndexError:
        del full[category]
        del state[category]
        continue

      state[category] -= 1
      sub = []
      
      if not self.D[category].exists(path):
        # replace path head with target path
        a, b = os.path.split(path)
        if self.which == 'cars':
          path = join(target_path, b)
        else:
          path = join(target_path, os.path.split(a)[1], b)
        sub.append((path, category))

      if len(sub):
        subs.append(sub)

    os.makedirs(submission_path, exist_ok=True)
    for i in range(sub_split):
      print(join(submission_path, f'{i}.txt'))
      with open(join(submission_path, f'{i}.txt'), 'w') as f:
        for sub in subs[i::sub_split]:
          for path, country in sub:
            f.write(f'{path},{country}\n')

def export_model(args):
  import subprocess
  PYTHON = sys.executable
  train_path = ('finetuning/train-4.py' if args.which == 'geo' else ('finetuning/train-cars.py' if args.which == 'cars' else 'finetuning/train-ftt.py'))
  if not os.path.exists(args.model_path.rstrip('/') + '-export'):
    path, checkpoint = os.path.split(args.model_path)
    subprocess.call([
        PYTHON,
        train_path,
        '--data_path', 'dataset/g3r',
        '--train_batch_size', '8',
        '--output_dir', path,
        '--resume_from_checkpoint', checkpoint,
        '--export-only',
        '--export-dir', args.model_path + '-export',
      ], env=os.environ.copy()
    )
  return args.model_path.rstrip('/') + '-export'

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-s', '--submission_path', required=True)
  parser.add_argument('-i', '--dataset_path', required=True)
  parser.add_argument('-t', '--target_path', required=False, default=None)
  parser.add_argument('-m', '--model_path', required=False, default=None)
  parser.add_argument('-c', '--typicality_path', required=True)
  parser.add_argument('--which', type=str, required=True, choices=['geo', 'ftt', 'cars'])
  parser.add_argument('--make_submission', action='store_true')

  parser.add_argument('--sub_split', type=int, default=1)
  parser.add_argument('--split_id', type=int, default=0)
  parser.add_argument('--t_min', type=float, default=0.1)
  parser.add_argument('--t_max', type=float, default=0.9)
  parser.add_argument('--dont_compute', action='store_false')
  parser.add_argument('--countries', nargs='*', default=None)
  args = parser.parse_args()

  model_path = args.model_path
  if args.model_path not in {'runwayml/stable-diffusion-v1-5', 'CompVis/stable-diffusion-v1-4'}:
    if not os.path.isfile(join(args.model_path, 'model_index.json')):
      model_path = export_model(args)

  if args.target_path is None:
    args.target_path = args.dataset_path

  mathieu = Typicality(args.which, model_path, args.dataset_path, args.typicality_path, t_min=args.t_min, t_max=args.t_max)
  if args.make_submission:
    mathieu.make_submission(args.target_path, args.submission_path, sub_split=args.sub_split)

  if args.dont_compute:
    assert args.model_path is not None
    mathieu.compute_submission(join(args.submission_path, str(args.split_id) + '.txt'))
