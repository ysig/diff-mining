
import PIL
import os
from os.path import join
import torch
import pandas as pd
import joblib
import numpy as np
from PIL import Image, ImageColor
from torch.nn.functional import interpolate
from scipy import ndimage
from scipy.ndimage import gaussian_filter

def normalize(dm, positive_only=False):
    # Standardize dm
    if positive_only:
      dm = np.maximum(dm, 0)

    dm = dm/np.max(np.abs(dm))
    return dm

def hcat(pils):
  width, height = pils[0].size
  total_width = sum(pil.width for pil in pils)
  new_im = Image.new(pils[0].mode, (total_width, height))
  x_offset = 0
  for im in pils:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]
  return new_im

def hcat_rgba_border(pils, border=1):
  width, height = pils[0].size
  total_width = sum(pil.width for pil in pils) + (len(pils) - 1) * border

  new_im = Image.new(pils[0].mode, (total_width, height), (0, 0, 0, 0))
  x_offset = 0
  for im in pils:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]
    if x_offset + border > total_width:
      break
    new_im.paste(Image.new('RGBA', (border, height), (0, 0, 0, 0)), (x_offset, 0))
    x_offset += border

  return new_im

def vcat(pils, vertical_spacing=0):
    width, _ = pils[0].size
    total_height = sum(pil.height for pil in pils) + vertical_spacing * (len(pils) - 1)
    new_im = Image.new(pils[0].mode, (width, total_height))
    y_offset = 0
    for idx, im in enumerate(pils):
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1] + (vertical_spacing if idx < len(pils) - 1 else 0)
    return new_im

# write a function that adds a square border around a pil image
def add_border(pil, color, border=1):
  # support colors by names
  pil = pil.convert('RGBA')
  if color == 'transparent':
    color = (0, 0, 0, 0)

  if isinstance(color, str):
    color = ImageColor.getrgb(color) + (255,)

  # allow to add transparent border with the string 'transparent'
  width, height = pil.size
  new_width, new_height = width + 2*border, height + 2*border
  result = Image.new(pil.mode, (new_width, new_height), color)
  result.paste(pil, (border, border))
  return result

def pool(losses, kx=5, ky=5):
  if kx != 1 and ky != 1:
    A, B, H, W = losses.shape
    losses = losses.reshape(A*B, 1, H, W)
    losses = torch.nn.AvgPool2d((kx, ky), stride=(1, 1), padding=0)(losses)
    losses = losses.reshape(A, B, losses.shape[-2], losses.shape[-1])
  return losses

def sort(df, key, ascending=False):
  return df.sort_values(by=[key], ascending=ascending).reset_index(drop=True)

def get_non_overlapping(df, k_per_image=5, merge_close_boxes=False):
  non_overlapping = []
  while len(non_overlapping) < k_per_image:
    non_overlapping.append(df.iloc[0])
    df = df[~((df['x_start'] <= non_overlapping[-1]['x_end']) & (df['x_end'] >= non_overlapping[-1]['x_start']) & (df['y_start'] <= non_overlapping[-1]['y_end']) & (df['y_end'] >= non_overlapping[-1]['y_start']))].reset_index(drop=True)
    if df.shape[0] == 0:
      break

  return pd.DataFrame(non_overlapping, columns=df.columns)

def load_image(x, resize_func=lambda x: x):
  pil = resize_func(PIL.Image.open(x['seed']))/255.0
  if 'x_start' not in x.columns:
    x_start, y_start, x_end, y_end = 0, 0, pil.size[1], pil.size[0]
  else:
    x_start, y_start, x_end, y_end = x[['x_start', 'y_start', 'x_end', 'y_end']]

  return pil.crop((y_start, x_start, y_end, x_end))

def get_top_k(df, k_per_image=5, k=1000, key='V', randomize=False, ascending=False, filter_by=[]):
  k = min(len(df), k)
  if randomize:
    return df.sample(k)
  elif len(filter_by) > 0:
    assert all(f in ['contrast', 'gradient'] for f in filter_by)
    filters = [((filter_by_contrast if f == 'contrast' else filter_by_gradient), karg) for f, karg in filter_by]
    total_elements = []
    for i in range(len(df)):
      pil = load_image(df.iloc[i])
      if all(f(pil, **karg) for f, karg in filters):
        total_elements.append(df.iloc[i])
    df = pd.DataFrame(total_elements, columns=df.columns)
  else:
    df = sort(df, key=key, ascending=ascending)
    return df.iloc[:k]

def make_grid(images, horizontal_spacing=2, vertical_spacing=4):
    if not images:
        return None
    
    # Assuming all images are of the same size
    img_width, img_height = images[0][0].size
    
    # Calculate total grid size
    max_len_w = max(len(i) for i in images)
    grid_width = img_width * max_len_w + horizontal_spacing * (max_len_w - 1)
    grid_height = img_height * len(images) + vertical_spacing * (len(images) - 1)
    
    # Create a new image with white background
    grid_image = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))
    
    # Paste images into the grid
    for row_idx, row in enumerate(images):
        for col_idx, img in enumerate(row):
            x_offset = col_idx * (img_width + horizontal_spacing)
            y_offset = row_idx * (img_height + vertical_spacing)
            grid_image.paste(img, (x_offset, y_offset))
    
    return grid_image
