import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

import subprocess
import math
import bisect
import PIL
import cv2
import numpy as np
import operator
import pandas as pd
import torch
import torchvision
import random
import copy
import argparse
import joblib
import json
from joblib import Parallel, delayed

from collections import defaultdict
from tqdm import tqdm, trange
from os.path import join
from PIL import Image
from matplotlib.cm import viridis
from matplotlib.colors import Normalize

from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
from torch.nn.functional import mse_loss
from transformers import CLIPTokenizer, CLIPTextModel
from torchvision.transforms import CenterCrop
from torchvision.transforms.functional import to_tensor
from torch.nn.functional import interpolate
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
from diffmining.typicality.utils import get_non_overlapping, sort, pool


class Embed(object):
  def __init__(self, clip, tokenizer, device):
    super().__init__()
    self.clip = clip
    self.tokenizer = tokenizer
    self.device = device

  @torch.no_grad()
  def tokenize(self, prompts):
      return self.tokenizer(prompts, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids

  @torch.no_grad()
  def embed_diseases(self, diseases):
    txt = [(f"Chest X-Ray with {c}." if len(c) else "Chest X-Ray") for c in diseases]
    tokens = self.tokenize(txt)
    return self.clip(tokens.to(self.device))[0].float()

  def __getitem__(self, x):
    return self.embed_diseases(x)

class SD(object):
  def __init__(self, model_path, diseases, device):
    self.device = device
    self.model = StableDiffusionPipeline.from_pretrained(
      model_path,
      tokenizer=CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14-336', torch_dtype=torch.float16),
      text_encoder=CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14-336', torch_dtype=torch.float16),
      torch_dtype=torch.float16,
    ).to(self.device)
    # self.model.enable_xformers_memory_efficient_attention()
    self.unet, self.vae, self.clip, self.clip_tokenizer, self.scheduler = self.model.unet.eval(), self.model.vae.eval(), self.model.text_encoder.eval(), self.model.tokenizer, self.model.scheduler

    self.text_features = Embed(self.clip, self.clip_tokenizer, self.device)
    self.diseases = sorted(diseases)
    apply_diseases = ["no finding"] + [""] + self.diseases
    cf = self.text_features[apply_diseases]
    self.country_embeds = {c: cf[i] for i, c in enumerate(apply_diseases)}

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
  def __init__(self, sd, seed=42, N=100):
    self.sd = sd
    self.seed = seed
    self.N = N

  @torch.no_grad()
  def noising(self, x):
    noise = torch.randn_like(x)
    timesteps = torch.randint(0, self.sd.scheduler.num_train_timesteps, (1,), device=self.sd.device)
    timesteps = timesteps.long()
    return noise, timesteps
  
  def load_image(self, x):
    x = x.convert('RGB')
    # to tensor
    with torch.autocast('cuda'):
      x = to_tensor(x)
      # normalize between -1 and 1
      x = x * 2 - 1
      x = x.unsqueeze(0)
    return x

  @torch.no_grad()
  def compute_losses(self, img, country_embeds, B=10):
    with torch.inference_mode():
      x = self.sd.encode_vae(self.load_image(img))

      torch.manual_seed(self.seed)
      noises, timesteps = zip(*[self.noising(x) for _ in range(self.N)])
      noises, timesteps = torch.cat(noises, dim=0), torch.cat(timesteps, dim=0)

      losses_grid = []
      n_diseases = country_embeds.size(0)
      for i in range(0, noises.shape[0], B):
        n_batch, t_batch = noises[i:i+B].to(self.sd.device), timesteps[i:i+B].to(self.sd.device)
        batch_size = n_batch.size(0)

        # stretch n_batch and t_batch to the number of diseases
        n_batch = torch.cat([n_batch]*n_diseases, dim=0)
        t_batch = torch.cat([t_batch]*n_diseases, dim=0)
        loss_grid = self.sd.compute_loss(x, n_batch, t_batch, torch.cat([country_embeds[c].unsqueeze(0).expand(batch_size, -1, -1) for c in range(n_diseases)], dim=0))

        # chunk so that the second dimension is the number of diseases
        loss_grid = torch.stack(torch.split(loss_grid, [batch_size]*n_diseases, dim=0), dim=1)
        losses_grid.append(loss_grid.cpu())
        
      losses_grid = torch.cat(losses_grid, dim=0)

    return losses_grid.to(dtype=torch.float16)

  def get_path(self, path):
    return join(self.typicallity_path, os.path.split(path)[1].replace('.jpg', '.npy'))

  def compute(self, country, path):
      img = PIL.Image.open(path)
      seed = os.path.split(path)[1]
      cfs = [self.sd.country_embeds[""]] #[self.sd.country_embeds["no finding"]], self.sd.country_embeds[""]]
      country_embeds = torch.stack([self.sd.country_embeds[country]] + cfs, dim=0)
      return self.compute_losses(img, country_embeds), img

  def exists(self, path):
    path = self.get_path(path)
    return os.path.isfile(path)

class Typicallity(object):
  def __init__(self, model_path, gt_path, output_path, diseases, seed=42):
    self.diseases = diseases
    self.output_path = output_path
    self.seed = seed
    self.load_paths(gt_path)
    self.gblur = torchvision.transforms.GaussianBlur((32*4-1, 32*4-1), sigma=(32, 32))
    self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    self.sd = SD(model_path, diseases, self.device)
    self.D = {country: D(self.sd) for country in self.parent.keys()}

  def load_paths(self, parent_path):
    df = pd.read_csv(join(parent_path, 'metadata.csv'))
    df = df[['Image Index', 'Finding Labels']]
    # rename columns to fname and label
    df.columns = ['fname', 'label']
    self.length = len(df)
    self.df = df
    self.image_folder = join(parent_path, 'images')
    
    bbox_df = pd.read_csv(join(parent_path, 'BBox_List_2017.csv'))
    bbox = {}
    # columns Image Index,Finding Label,Bbox [x,y,w,h],,,
    bbox_df.columns = ['fname', 'label', 'bbox-x', 'bbox-y', 'bbox-w', 'bbox-h']
    for i, row in bbox_df.iterrows():
      bbox[(row['fname'], row['label'])] = (row['bbox-x'], row['bbox-y'], row['bbox-x'] + row['bbox-w'], row['bbox-y'] + row['bbox-h'])
      # divide by 2
      bbox[(row['fname'], row['label'])] = tuple(map(lambda x: int(x/2), bbox[(row['fname'], row['label'])]))    
    
    # keep from self.df only fnames that are in bbox
    fnames = set(bbox_df['fname'])
    self.df = self.df[self.df['fname'].isin(fnames)]
    print(f"Number of images: {len(self.df)}")

    self.parent = defaultdict(list)
    for i, row in self.df.iterrows():
      label = row['label']
      diseases = label.split('|')
      for disease in self.diseases:
        if disease in diseases:
          if (row['fname'], disease) in bbox:
            self.parent[disease].append((join(self.image_folder, row['fname']), diseases, bbox[(row['fname'], disease)]))

    random.seed(self.seed)
    for k, v in self.parent.items():
      self.parent[k] = sorted(v, key=lambda x: (len(x[1]), random.random()))
      self.parent[k] = [(a, c) for a, _, c in self.parent[k]]

  def blur(self, dm, i):
    return (self.gblur(dm[:, -1].unsqueeze(1)) - self.gblur(dm[:, i].unsqueeze(1))).squeeze(1).mean(axis=0).cpu().numpy()

  @torch.no_grad()
  def compute(self, dm, size, blur=False):
    dm = dm.float()
    dm = dm.mean(dim=2)
    dm = interpolate(dm, size, mode="bilinear")
    dm_pixel = (dm[:, -1] - dm[:, 0]).mean(dim=0).cpu().numpy()
    if blur:
      dm = self.blur(dm, 0)
    return dm, dm_pixel

  def predict_bboxes(self, dm, k_per_image=5, ascending=True):
      df = [(i, j, i+self.kx, j+self.ky, dm[i, j]) for i in range(dm.shape[0]) for j in range(dm.shape[1])]
      df = pd.DataFrame(df, columns=['x_start', 'y_start', 'x_end', 'y_end', 'D'])
      df = sort(df, 'D', ascending=ascending)

      return get_non_overlapping(df, k_per_image=k_per_image)

  def visualize_boxes(self, gt_box, dm, pil):
    pil = pil.convert('RGB')

    # convert to opencv
    img = np.array(pil)

    # Standardize dm
    dm_mean = np.mean(dm)
    dm_std = np.std(dm)
    dm_standardized = (dm - dm_mean) / dm_std

    # Normalize dm to range 0 to 1
    dm_min = np.min(dm_standardized)
    dm_max = np.max(dm_standardized)
    dm_normalized = (dm_standardized - dm_min) / (dm_max - dm_min)

    # Apply viridis colormap to dm_normalized and convert to RGBA
    norm = Normalize(vmin=0, vmax=1)
    dm_colored = (viridis(norm(dm_normalized)) * 255).astype(np.uint8)

    # Overlay dm_colored_resized over the original image
    overlayed_img = img.copy()
    overlay_mask = dm_colored[:, :, 3] > 0  # Mask where alpha > 0
    for i in range(3):  # RGB channels
      overlayed_img[overlay_mask, i] = (
        overlayed_img[overlay_mask, i].astype(float) * (1 - dm_colored[overlay_mask, 3].astype(float) / 255 * 0.7) +
        dm_colored[overlay_mask, i].astype(float) * (dm_colored[overlay_mask, 3].astype(float) / 255 * 0.7)
      ).astype(np.uint8)
    x1, y1, x2, y2 = gt_box
    cv2.rectangle(overlayed_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # horizontal concat to the original image
    img = np.concatenate([img, overlayed_img], axis=1)

    return img

  def mean_typicallity(self, bbox, dm):
    return dm[bbox[1]:bbox[3], bbox[0]:bbox[2]].mean()

  def aucpr(self, bbox, dm):
    # explore all values between the range of 0.1 and 0.000006
    thresholds = 2*10**(-np.linspace(2, 7, 1000))

    # compute the true positive rate and false positive rate
    x = np.zeros_like(dm)
    x[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
    dm_flattened = dm.flatten()
    x_flattened = x.flatten()
    tp = np.sum(dm_flattened[x_flattened == 1] > thresholds[:, np.newaxis], axis=1)
    fp = np.sum(dm_flattened[x_flattened == 0] > thresholds[:, np.newaxis], axis=1)

    # compute the precision and recall
    denominator = tp + fp
    precision = np.where(denominator > 0, tp / denominator, 0)
    recall = tp / x.sum()

    # compute the area under the precision-recall curve
    return np.trapz(precision, recall)

  def main(self, num_per_disease=10):
    report, auc = {}, {}
    total = sum(1 for disease in self.diseases for _ in self.parent[disease])
    idx = 0
    pbar = tqdm(total=total)
    for disease in self.diseases:
      report[disease], auc[disease] = {}, {}
      os.makedirs(join(self.output_path, disease, 'typicality'), exist_ok=True)
      for fpath, bbox in self.parent[disease]:
        name = os.path.split(fpath)[-1].replace('.jpg', '').replace('.png', '')
        if os.path.isfile(join(self.output_path, disease, 'typicality', f"{name}_loss_pixel.npy")):
          loss_pixel = np.load(join(self.output_path, disease, 'typicality', f"{name}_loss_pixel.npy"))
        else:
          os.makedirs(join(self.output_path, disease), exist_ok=True)
          loss, pil = self.D[disease].compute(disease, fpath)
          loss, loss_pixel = self.compute(loss, pil.size)
          
          # Save loss and loss pixel to disk
          # np.save(join(self.output_path, disease, 'typicality', f"{name}_loss.npy"), loss)
          np.save(join(self.output_path, disease, 'typicality', f"{name}_loss_pixel.npy"), loss_pixel)
        
        report[disease][os.path.split(fpath)[-1]] = float(self.mean_typicallity(bbox, loss_pixel))
        auc[disease][os.path.split(fpath)[-1]] = float(self.aucpr(bbox, loss_pixel))
        # img = self.visualize_boxes(bbox, loss, pil)

        # img = Image.fromarray(img)
        # img.save(join(self.output_path, disease, os.path.split(fpath)[1]))
        pbar.update(1)
        # idx += 1
        # if idx % 100 == 0:
        #   # save file to self.output_path
        #   with open(join(self.output_path, 'report.json'), 'w') as f:
        #     json.dump(report, f, indent=4)

      if not len(report[disease]):
        del report[disease]
        del auc[disease]

    # save file to self.output_path
    with open(join(self.output_path, 'report.json'), 'w') as f:
      json.dump(report, f, indent=4)

    # save file to self.output_path
    with open(join(self.output_path, 'auc.json'), 'w') as f:
      json.dump(auc, f, indent=4)

    return self

def export_model(args):
  PYTHON = sys.executable
  if not os.path.exists(args.model_path.rstrip('/') + '-export'):
    path, checkpoint = os.path.split(args.model_path)
    subprocess.call([
        PYTHON,
        'train.py',
        '--data_path', 'dataset/CXR8',
        '--train_batch_size', '8',
        '--output_dir', path,
        '--resume_from_checkpoint', checkpoint,
        '--export-only',
        '--export-dir', args.model_path + '-export',
      ], env=os.environ.copy()
    )

def compare_json_files(json_pt, json_ft):
  with open(join(json_pt, 'auc.json'), 'r') as f:
    data_pt = json.load(f)

  with open(join(json_ft, 'auc.json'), 'r') as f:
    data_ft = json.load(f)

  print('AUC\n----------')
  for k, vs in data_pt.items():
    print('ft', k, np.mean([data_ft[k][kp] for kp, v in vs.items()]), '±', np.std([data_ft[k][kp] for kp, v in vs.items()]))
    print('pt', k, np.mean([data_pt[k][kp] for kp, v in vs.items()]), '±', np.std([data_pt[k][kp] for kp, v in vs.items()]))
    print(k, np.mean([data_ft[k][kp] - data_pt[k][kp] for kp, v in vs.items()]))


  # Extract the values for the stripplot
  df = []
  for k, vs in data_pt.items():
    df.extend([{'model': 'pt', 'disease': k, 'score': data_pt[k][kp]} for kp, v in vs.items()])
    df.extend([{'model': 'ft', 'disease': k, 'score': data_ft[k][kp]} for kp, v in vs.items()])

  import seaborn as sns
  import pandas as pd
  import matplotlib.pyplot as plt
  df = pd.DataFrame(df)
  sns.stripplot(x='disease', y='score', data=df, hue='model', jitter=0.2, dodge=True)
  plt.xlabel('Model')
  plt.ylabel('Value')
  plt.title('Comparison of Values between pt and ft')
  plt.savefig('comparison2.png')

  with open(join(json_pt, 'report.json'), 'r') as f:
    data_pt = json.load(f)

  with open(join(json_ft, 'report.json'), 'r') as f:
    data_ft = json.load(f)

  print('Typicality\n----------')
  for k, vs in data_pt.items():
    print('ft', k, np.mean([data_ft[k][kp] for kp, v in vs.items()]), '±', np.std([data_ft[k][kp] for kp, v in vs.items()]))
    print('pt', k, np.mean([data_pt[k][kp] for kp, v in vs.items()]), '±', np.std([data_pt[k][kp] for kp, v in vs.items()]))
      

def merge_triplets(pt, ft, data_path, triplet_path):
  os.makedirs(triplet_path, exist_ok=True)
  total = sum(1 for disease in os.listdir(pt) for image in os.listdir(join(pt, disease)))
  pbar = tqdm(total=total)
  for disease in os.listdir(pt):
    os.makedirs(join(triplet_path, disease), exist_ok=True)
    if disease not in {'auc.json', 'report.json'}:
      for image in os.listdir(join(pt, disease)):
        img_pt = PIL.Image.open(join(pt, disease, image))
        img_ft = PIL.Image.open(join(ft, disease, image))
        img_data = PIL.Image.open(join(data_path, 'images', image))
        img = PIL.Image.new('RGB', (img_pt.width//2, img_pt.height * 3))
        img.paste(img_data, (0, 0))
        img.paste(img_pt.crop((512, 0, 1024, 512)), (0, img_pt.height))
        img.paste(img_ft.crop((512, 0, 1024, 512)), (0, img_pt.height*2))
        img.save(join(triplet_path, disease, image))
        pbar.update(1)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--gt_path', type=str, required=False, default='dataset/CXR8')
  parser.add_argument('-o', '--output_path', type=str, required=False, default='results/ct')
  parser.add_argument('-m', '--model_path', type=str, required=False, default='models/CXR8')
  args = parser.parse_args()

  model_path = args.model_path
  if args.model_path not in {'runwayml/stable-diffusion-v1-5', 'CompVis/stable-diffusion-v1-4'}:
    if not os.path.isfile(join(args.model_path, 'model_index.json')):
      export_model(args)
      model_path = args.model_path.rstrip('/') + '-export'

  diseases = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltrate', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']
  # typicallity = Typicallity(model_path, args.gt_path, args.output_path, diseases).main()
  compare_json_files('/home/isig/diff-mining/results-fix-new/pt', '/home/isig/diff-mining/results-fix-new/ft')