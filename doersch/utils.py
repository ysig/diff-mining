import numpy as np
import random
import skimage
from PIL import Image

def hcat(pils, margin=0):
  width, height = pils[0].size
  total_width = sum(pil.width for pil in pils) + (margin * (len(pils) - 1))
  new_im = Image.new(pils[0].mode, (total_width, height), (255, 255, 255))
  x_offset = 0
  for im in pils:
    new_im.paste(im, (x_offset, 0))
    x_offset += im.size[0] + margin
  return new_im

def add_border(image, margin, color):
  width, height = image.size
  new_width = width + 2 * margin
  new_height = height + 2 * margin
  new_image = Image.new(image.mode, (new_width, new_height), color)
  new_image.paste(image, (margin, margin))
  return new_image

def vcat(pils, margin=0):
  width = max(pil.width for pil in pils)
  total_height = sum(pil.height for pil in pils) + (margin * (len(pils) - 1))
  new_im = Image.new(pils[0].mode, (width, total_height), (255, 255, 255))
  y_offset = 0
  for im in pils:
    new_im.paste(im, (0, y_offset))
    y_offset += im.size[1] + margin
  return new_im

# def filter_by_contrast(patch, threshold):
#   pixel_values = np.array(patch).flatten()
#   contrast = pixel_values.std()
#   return contrast > threshold
def filter_by_contrast(patch, threshold):
  return not skimage.exposure.is_low_contrast(np.array(patch), fraction_threshold=0.15, lower_percentile=1, upper_percentile=99, method='linear')

def overlap(bbox_list, bbox):
  for b in bbox_list:
    if b[0] < bbox[2] and bbox[0] < b[2] and b[1] < bbox[3] and bbox[1] < b[3]:
      return True
  return False

def seed_all(self, seed):
  np.random.seed(seed)
  random.seed(seed)

def iou(bbox_a, bbox_b):
  x1 = max(bbox_a[0], bbox_b[0])
  y1 = max(bbox_a[1], bbox_b[1])
  x2 = min(bbox_a[2], bbox_b[2])
  y2 = min(bbox_a[3], bbox_b[3])
  inter = max(0, x2 - x1) * max(0, y2 - y1)
  area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
  area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
  return inter / (area_a + area_b - inter)