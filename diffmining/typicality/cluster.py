import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import operator
import random
import copy
import math
import argparse
import joblib
import PIL

from PIL import Image
from joblib import Parallel, delayed
from collections import defaultdict
from os.path import join
from tqdm.auto import tqdm, trange

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from scipy import ndimage
from scipy.ndimage import gaussian_filter

import torch
from torch.nn.functional import interpolate
from diffmining.typicality.utils import sort, pool, get_top_k, get_non_overlapping, hcat_rgba_border, add_border, normalize, filter_patch, hcat_margin, apply_alpha, make_grid
from diffmining.typicality.dift import SDFeaturizer, dift_pre
from diffmining.typicality.compute import Typicality

def normalize(dm, positive_only=False):
    # Standardize dm
    if positive_only == 'split':
      dm = dm/np.abs(np.max(dm))
      return np.clip(dm, 0, 1), -np.clip(dm, -1, 0)

    elif positive_only:
      dm = np.maximum(dm, 0)
      dm_max = np.max(dm)
      dm_normalized = dm / dm_max
    else:
      # rescale positives by positive deviation and negatives by negative deviation
      dm[dm < 0] = dm[dm < 0]/np.abs(np.min(dm))
      dm[dm > 0] = dm[dm > 0]/np.max(dm)
      dm_normalized = (dm + 1)/2.0

    return dm_normalized

def mean(vs):
  return sum(v[1] for v in vs)/(1.0*len(vs))

def median(vs):
  return np.median([v[1] for v in vs])

class Cluster(Typicality):
  def __init__(self, which, typicality_path, dataset_path, cache_path, recache, model_path, aggregate='median', kx=64, ky=64, cache_features=True):
    super().__init__(which=which, model_path=None, dataset_path=dataset_path, typicality_path=typicality_path, xformers=False)
    self.cache_path = cache_path
    self.recache = recache
    self.kx = kx
    self.ky = ky
    self.model_path = model_path
    self.aggregate = (mean, median)[int(aggregate == 'median')]
    self.cache_features = cache_features

  def get_seeds(self, d, tag):
    if self.which in {'ftt', 'cars'}:
      return [path for path in self.times[tag] if d.exists(path)]
    elif self.which == 'geo':
      return [path[0] for path in self.country_path[tag] if path[1] and d.exists(path[0])]
    elif self.which == 'places':
      return [path for path in self.parent[tag] if d.exists(path)]

  def load_image(self, path, pil=True):
    img = PIL.Image.open(path).convert('RGB')
    if self.which == 'cars':
      if img.width > img.height:
        img = img.resize((int(img.width * (256 / img.height)), 256), Image.LANCZOS)
      else:
        img = img.resize((256, int(img.height * (256 / img.width))), Image.LANCZOS)
    elif self.which == 'places':
      if img.width > img.height:
          img = img.resize((math.ceil(img.width * (512 / img.height)), 512), Image.LANCZOS)
      else:
          img = img.resize((512, math.ceil(img.height * (512 / img.width))), Image.LANCZOS)

    if pil:
      return img
    else:
      return np.array(img)/255.0

  def load_and_apply_alpha_bbox(self, country, pil, idd, e):
    # get the bbox of pild
    bbox, country, path = self.decompose_save_path(idd, e, country)
    alpha = self.load_typicality_norm(self.D[country], path)

    # crop alpha to bbox
    I = self.load_image(path, pil=False)
    T = gaussian_filter(alpha, sigma=10)
    T = T*(T>0)
    T = np.stack((T,T,T), axis=-1)

    # pad T with zeros
    a = np.zeros_like(I)
    a[:T.shape[0], :T.shape[1], :] = T
    T = a
    R = 0.05*I + 0.95*(T*I + (1-T))
    R = R[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    return PIL.Image.fromarray((R*255).astype(np.uint8)), a[bbox[0]:bbox[2], bbox[1]:bbox[3]]

  def load_typicality_norm(self, d, path):
    w, h = self.load_image(path).size
    dm = d(path)
    dm = torch.from_numpy(dm).to(self.device)
    # convert from float16 to float32
    dm = dm.float()
    dm = dm.mean(dim=2)
    # print(dm.dtype, dm.device)
    dm = interpolate(dm, (h, w), mode="bilinear")
    dm = (dm[:, 1] - dm[:, 0]).mean(dim=0).cpu().numpy()
    dm = normalize(dm)
    return dm

  @torch.no_grad()
  def load_typicality(self, d, path):
    w, h = self.load_image(path).size
    dm = d(path)
    dm = torch.from_numpy(dm).to(self.device)
    # convert from float16 to float32
    dm = dm.float()
    dm = dm.mean(dim=2)
    # print(dm.dtype, dm.device)
    dm = interpolate(dm, (h, w), mode="bilinear")
    dm = pool(dm[:, 0].unsqueeze(1), self.kx, self.ky) - pool(dm[:, 1].unsqueeze(1), self.kx, self.ky)
    dm = -dm.squeeze(1).mean(dim=0).cpu().numpy()
    return dm

  def decompose_save_path(self, idd, path, country):
    bbox = idd.split('_')
    x_start, y_start, x_end, y_end = bbox[-1].split('-')
    return (int(x_start), int(y_start), int(x_end), int(y_end)), country, path

  def copy_to_images_dir(self, html_path, pil, tag):
    os.makedirs(join(html_path, 'clusters', 'images'), exist_ok=True)
    path_ = join(html_path, 'clusters', 'images', tag + '.png')
    if not os.path.isfile(path_):
      pil.save(path_)
    return join('images', tag + '.png')

  def apply_alpha_(self, pil_path):
    parent, path = os.path.split(pil_path)
    ext = os.path.splitext(path)[1]
    path_alpha = join(parent, 'alpha-' + path.replace(ext, '.pkl'))
    ext = os.path.splitext(path)[1]
    pil = PIL.Image.open(pil_path).convert('RGB') 
    I = np.array(pil)/255.0
    T = joblib.load(path_alpha)
    R = 0.05*I + 0.95*(T*I + (1-T))
    return PIL.Image.fromarray((R*255).astype(np.uint8))

  def resize(self, pil):
    if self.which == 'cars':
      w, h = pil.size
      if w > h:
        w = int(w * 256 / h)
        h = 256
      else:
        h = int(h * 256 / w)
        w = 256
      pil = pil.resize((w, h), PIL.Image.LANCZOS)
    elif self.which == 'places':
      w, h = pil.size
      if w > h:
        w = int(w * 512 / h)
        h = 512
      else:
        h = int(h * 512 / w)
        w = 512
      pil = pil.resize((w, h), PIL.Image.LANCZOS)
    return pil

  @torch.no_grad()
  def df_D(self, country, k_per_image, seed=42, n_jobs=12, ascending=False, gt_only=False):
    d = self.D[country]

    # parallel
    def compute(path):
      try:
        dm = self.load_typicality(d, path)

        # take the top-5
        kx, ky = self.kx, self.ky
        df = [(path, i, j, i+kx, j+ky, dm[i, j], 'real') for i in range(dm.shape[0]) for j in range(dm.shape[1])]
        
        df_random = copy.deepcopy(df)
        random.shuffle(df_random)
        df_random = pd.DataFrame(df_random, columns=['seed', 'x_start', 'y_start', 'x_end', 'y_end', 'D', 'origin'])

        df = pd.DataFrame(df, columns=['seed', 'x_start', 'y_start', 'x_end', 'y_end', 'D', 'origin'])
        df = sort(df, 'D', ascending=ascending)

        torch.cuda.empty_cache()
        return get_non_overlapping(df, k_per_image=k_per_image), get_non_overlapping(df_random, k_per_image=k_per_image)
      except Exception as ex:
        import traceback
        traceback.print_exc()
        print('error', ex)
        print('@path=', path)
        return pd.DataFrame([], columns=['seed', 'x_start', 'y_start', 'x_end', 'y_end', 'D', 'origin']), pd.DataFrame([], columns=['seed', 'x_start', 'y_start', 'x_end', 'y_end', 'D', 'origin'])

    paths = self.get_seeds(d, country)
    parallel = Parallel(n_jobs=n_jobs, timeout=100000)(delayed(compute)(path) for path in tqdm(paths, desc=f'Extracting D [{country}]'))
    topk, randomized = zip(*list(parallel))
    return pd.concat([df for df in topk], axis=0), pd.concat([df for df in randomized], axis=0)

  def init__clip(self, clip_model="openai/clip-vit-base-patch32"):
    if not hasattr(self, 'clip'):
      from transformers import CLIPProcessor, CLIPModel
      self.processor = CLIPProcessor.from_pretrained(clip_model)
      self.clip_model = CLIPModel.from_pretrained(clip_model).to(self.device)

  @torch.no_grad()
  def embed(self, a, prompt=None, t=None):
    if t is not None:
      return self.dift.forward(dift_pre(a), t=t, prompt=prompt, ensemble_size=8).squeeze(0).cpu().numpy()
    else:
      image_input = self.processor(images=[a], return_tensors="pt")
      features = self.clip_model.get_image_features(**image_input.to(self.device))
      features /= features.norm(dim=-1, keepdim=True)
      return features.squeeze(0).cpu().numpy()

  def dift_prompt(self, c):
    if self.which == 'cars':
      return f"Portrait at the {c}'s." if len(c) else "Portrait."
    elif self.which == 'faces':
      return f"A car at the {c}'s." if len(c) else "A car."
    elif self.which == 'places':
      return ('Image of ' + c.replace('_', ' ') + '.'  if len(c) else "")
    else:
      return f"{c}" if len(c) else ""

  def compute_embeddings(self, df, c, to_add_border=True, feature_which='dift-261',):
    self.init__clip()
    X, ids, pils, ds, orig_path = [], [], [], [], []
    
    dift = 'dift' in feature_which
    clip = 'clip' in feature_which

    if clip and dift:
      t = int(feature_which.split('+')[1].split('-')[1])
    elif dift:
      t = int(feature_which.split('-')[1])

    for i in trange(df.shape[0]):
      pil = self.resize(PIL.Image.open(df.iloc[i]['seed']))

      if 'x_start' not in df.columns:
        x_start, y_start, x_end, y_end = 0, 0, pil.size[1], pil.size[0]
      else:
        x_start, y_start, x_end, y_end = df.iloc[i][['x_start', 'y_start', 'x_end', 'y_end']]
      pil = pil.crop((y_start, x_start, y_end, x_end))

      ds.append(df.iloc[i]['D'])
      orig_path.append(df.iloc[i]['seed'])

      # in this case we put red if element is ground truth and transparent otherwise
      path__ = os.path.split(df.iloc[i]['seed'])[1]
      ext = os.path.splitext(path__)[1]
      ids.append(path__.replace(ext, '_') + f'{x_start}-{y_start}-{x_end}-{y_end}')
      pils.append(pil)

      pkl_file = join(self.cache_path, 'embeddings', feature_which, ids[-1] + '.pkl')
      if self.cache_features and os.path.isfile(pkl_file):
        emb = joblib.load(pkl_file)
      else:
        image = self.load_image(df.iloc[i]['seed'])
        if clip and dift:
          emb_a = self.embed(image.crop((y_start, x_start, y_end, x_end)).convert('RGB'))
          emb_b = self.embed(image.convert('RGB'), prompt=self.dift_prompt(c), t=t)

          _, h, w = emb_b.shape
          H = h/image.height
          W = w/image.width

          emb_b = emb_b[:, int(x_start*H):int(x_end*H), int(y_start*W):int(y_end*W)]
          emb_b = emb_b.mean(axis=(1, 2))
          emb_b = emb_b / np.linalg.norm(emb_b)

          emb = np.concatenate([emb_a, emb_b])
        elif dift:
          emb = self.embed(image.convert('RGB'), prompt=self.dift_prompt(c), t=t)
          _, h, w = emb.shape
          H = h/image.height
          W = w/image.width

          emb = emb[:, int(x_start*H):int(x_end*H), int(y_start*W):int(y_end*W)]
          emb = emb.mean(axis=(1, 2))
          emb = emb / np.linalg.norm(emb)
        else:
          emb = self.embed(image.convert('RGB').crop((y_start, x_start, y_end, x_end)))
        
        if self.cache_features:
          joblib.dump(emb, pkl_file)

      X.append(emb)
      if to_add_border:
        pils[-1] = add_border(pils[-1], 'transparent' if df.iloc[i]['origin'] == 'fake' else 'red')

    return X, ids, pils, ds, orig_path

  def cluster(self, X, ids, pils, ds, real_paths, country, num_clusters=8, project=True):
    clustering = KMeans(n_clusters=num_clusters, random_state=10)
    if project:
        umap_model = umap.UMAP(n_components=5)
        X = umap_model.fit_transform(X)
    clustering.fit(np.stack(X, axis=0))
    clusters = defaultdict(list)
    for pil, (i, l) in zip(pils, enumerate(clustering.labels_)):
      clusters[l].append((pil, ds[i], ids[i], X[i], real_paths[i]))

    # find the centroid of each cluster
    for k, vs in clusters.items():
      # find the point that is closer to the cluster center
      clusters[k] = [(a, b, c, e) for a, b, c, d, e in sorted(vs, key=lambda x: np.linalg.norm(x[-2] - clustering.cluster_centers_[k]))]

    # Assuming 'X' is the data and 'clustering.labels_' are the cluster labels
    return sorted([(vs, self.aggregate(vs), ) for _, vs in clusters.items()], key=operator.itemgetter(1), reverse=True)

  def clustering(self, feature_which, k_per_image=5, k=1000, num_clusters=32, only_gt=True, project=False):
    def copy_to_images_dir(pil, parent_, tag, local_dir):
      os.makedirs(parent_, exist_ok=True)
      if not os.path.isfile(tag):
        pil.save(join(parent_, tag))
      return join(local_dir, tag)

    dfs, dfs_random = {}, {}
    cache_path = join(self.cache_path, 'clusters')
    for country in tqdm(self.categories(), desc='clustering [extracting D]'):
      fp = os.path.join(cache_path, country + '.pkl')
      try:
        joblib.load(fp)
      except Exception:
        os.makedirs(cache_path, exist_ok=True)
        df, df_random = self.df_D(country, k_per_image=k_per_image, gt_only=only_gt)
        joblib.dump((df, df_random), fp)

    for country in tqdm(self.categories(), desc='clustering [extracting D]'):
      df, df_random = joblib.load(os.path.join(cache_path, country + '.pkl'))
      if only_gt:
        df = df[df.origin == 'real'].reset_index(drop=True)
        df_random = df_random[df_random.origin == 'real'].reset_index(drop=True)

      assert len(df)
      dfs[country] = get_top_k(df, key='D', k=k, randomize=False)
      dfs_random[country] = get_top_k(df_random, key='D', k=k, randomize=True)

    if 'dift' in feature_which:
      self.dift = SDFeaturizer(self.model_path, text_encoder_id=("geolocal/StreetCLIP" if self.which == 'geo' else "openai/clip-vit-large-patch14-336"))

    os.makedirs(join(self.cache_path, 'embeddings', feature_which), exist_ok=True)
    for randomize in [False] + ([True] if self.model_path in {'runwayml/stable-diffusion-v1-5', 'CompVis/stable-diffusion-v1-4'} else []):
      random_tag = ('ranked', 'random')[int(randomize)]
      embs = {country: self.compute_embeddings((dfs_random if randomize else dfs)[country], c=country, to_add_border=not only_gt, feature_which=feature_which) for country in tqdm(self.categories(), desc=f'clustering [embeddings]')}
      clusters = {country: self.cluster(*embs[country], num_clusters=num_clusters, country=country, project=project) for country in tqdm(self.categories(), desc=f'clustering [clustering]')}

      for country in sorted(self.categories()):
        local_dir = join('images', 'clusters', random_tag, feature_which, country)
        parent_ = join(self.cache_path, local_dir)
        os.makedirs(parent_, exist_ok=True)

        # column should be the cluster id
        for i in trange(num_clusters):
          for j, (pil, ds, idd, _) in enumerate(clusters[country][i][0]):
            pil.save(join(parent_, f'{i}-{j}-{num_clusters}_{idd}.png'))
            #pil_alpha, alpha_only = self.load_and_apply_alpha_bbox(country, pil, idd, _)
            # save alpha_only as pkl
            #alpha_only_path = join(parent_, f'alpha-{i}-{j}-{num_clusters}_{idd}.pkl')
            #pil_alpha.save(join(parent_, f'alpha-{i}-{j}-{num_clusters}_{idd}.png'))
            #joblib.dump(alpha_only, alpha_only_path)

  def compute_least(self, gt_only=True, k_per_image=5, n_jobs=12):
    tag_gt = ('-gt' if gt_only else '')
    dfs, dfs_random = {}, {}
    cache_path = join(self.cache_path, 'clusters')
    for country in tqdm(self.categories(), desc='clustering [extracting D]'):
      fp = os.path.join(cache_path, country + tag_gt + '_least.pkl')
      try:
        dfs[country], dfs_random[country] = joblib.load(fp)
      except Exception as ex:
        print(country, ex)
        os.makedirs(cache_path, exist_ok=True)
        dfs[country], dfs_random[country] = self.df_D(country, k_per_image=k_per_image, ascending=True, n_jobs=n_jobs, gt_only=True)
        joblib.dump((dfs[country], dfs_random[country]), fp)

    return dfs

  def plot_top_k(self, k_per_image=5, k=200):
    dfs, dfs_random = {}, {}
    for country in tqdm(self.categories(), desc='clustering [extracting D]'):
      fp = join(self.cache_path, 'clusters', country + '.pkl')
      if not os.path.isfile(fp) or self.recache:
        os.makedirs(join(self.cache_path, 'clusters'), exist_ok=True)
        df, df_random = self.df_D(country, k_per_image=k_per_image, gt_only=True)
        joblib.dump((df, df_random), fp)

      df, df_random = joblib.load(fp)
      dfs[country] = get_top_k(df, key='D', k=k, randomize=False)
      dfs_random[country] = get_top_k(df_random, key='D', k=k, randomize=True)

    # do the same for pd measure
    os.makedirs(join(self.cache_path, 'clusters'), exist_ok=True)
    dfs_least = self.compute_least()
    for c in tqdm(self.categories(), desc='clustering [extracting D least]'):
      dfs_least[c] = get_top_k(dfs_least[c], key='D', k=k, randomize=False, ascending=True)

    # save all 100 images in two horizontal arrays of 50 images each as pil images with margin 0.1
    for name, dfs_ in zip(['D', 'random', 'D_least'], [dfs, dfs_random, dfs_least]):
      # extract pil images
      pils = {}
      for country in self.categories():
        df = dfs_[country]
        pils_ = []
        for i in range(df.shape[0]):
          x_start, y_start, x_end, y_end = df.iloc[i][['x_start', 'y_start', 'x_end', 'y_end']]
          pils_.append(self.resize(PIL.Image.open(df.iloc[i][('path_' + str(df.iloc[i]['origin']) if 'bar' in name else 'seed')])).crop((y_start, x_start, y_end, x_end)).convert('RGBA'))
        pils[country] = pils_

      # save them
      os.makedirs(join(self.cache_path, 'images', 'topk', name), exist_ok=True)
      for c in self.categories():
        os.makedirs(join(self.cache_path, 'images', 'topk', name, c), exist_ok=True)
        for i in range(len(pils[c])):
          pils[c][i].save(join(self.cache_path, 'images', 'topk', name, c, str(i) + '.png'))

  def apply_alpha_(self, image_path):
    return apply_alpha(image_path)

  def make_figure(self, figure_path, hard_limit=6, top_k=5, min_im=5, feature_which=None, topk=False, grid_sep_x=2, grid_sep_y=2):
    # assumes that html has been already created
    # assumes that the top-k images have been already extracted
    for which in ['ranked', 'random']:
      dirr = join(self.cache_path, 'images', 'clusters')

      if '/ft' in figure_path and which == 'random':
        continue

      if not os.path.isdir(dirr):
        continue

      for feature_type in os.listdir(join(dirr, which)):
        if feature_which != 'all' and feature_type != feature_which:
          continue

        group = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
        for t in os.listdir(join(self.cache_path, 'images', 'clusters', which, feature_type)):
          parent_ = join(self.cache_path, 'images', 'clusters', which, feature_type, t)
          for image_path in os.listdir(parent_):
            if 'alpha' in image_path:
              continue

            image_path_ = join(parent_, image_path)
            cluster_id, idx, num_clusters = image_path.split('-')[:3]
            num_clusters = num_clusters.split('_')[0]
            group[t][num_clusters][cluster_id].append((idx, image_path_))

        # collect the top-5 predictions that don't have less than 6 elements and put them in a grid with a spacing of 2 pixels per column and 4 pixels per line
        for t, a in group.items():
          for nc, b in a.items():
            print('nc', nc, 'top_k', top_k)
            images_grid, images_grid_alpha = [], []
            bs = list(b.items())
            for i, cluster_elems in sorted(bs, key=lambda x: int(x[0])):
              images = sorted(cluster_elems, key=lambda x: int(x[0]))
              if len(images_grid) == top_k:
                break
              if len(images) < min_im:
                continue

              images_grid.append([PIL.Image.open(image_path_).convert('RGB') for _, image_path_ in images[:hard_limit]])
              # images_grid_alpha.append([self.apply_alpha_(image_path_) for _, image_path_ in images[:hard_limit]])

            if len(images_grid):
              ending = f'{t}'
              if hard_limit != 6:
                ending += f'__hard_limit_{hard_limit}'
              if top_k != 5:
                ending += f'__top_k_{top_k}'
              if min_im != 5:
                ending += f'__min_im_{min_im}'

              os.makedirs(join(figure_path, 'clusters'), exist_ok=True)
              make_grid(images_grid, horizontal_spacing=grid_sep_x, vertical_spacing=grid_sep_y).save(join(figure_path, 'clusters', f'{ending}_{which}.png'))
              # make_grid(images_grid_alpha, horizontal_spacing=2, vertical_spacing=4).save(join(parent, f'{ending}_alpha.png'))

    #  top-k images for each cluster
    if topk:
      for name in os.listdir(join(self.cache_path, 'images', 'topk')):
        for c in os.listdir(join(self.cache_path, 'images', 'topk', name)):
          pils = []
          files = os.listdir(join(self.cache_path, 'images', 'topk', name, c))
          for file in sorted(files, key=lambda x: int(x.split('.')[0])):
            pil = PIL.Image.open(join(self.cache_path, 'images', 'topk', name, c, file))
            if filter_patch(pil):
              pils.append(pil)
              if len(pils) == 7:
                break

          os.makedirs(join(figure_path, 'topk', c), exist_ok=True)
          hcat_margin(pils).save(join(figure_path, 'topk', c, f'{name}.png'))

  @torch.no_grad()
  def rank_images(self, country, seed=42, n_jobs=12, ascending=False, gt_only=False):
    d = self.D[country]

    # parallel
    def compute(path):
      try:
        w, h = self.load_image(path).size
        dm = d(path[0])
        dm = torch.from_numpy(dm).to(self.device)
        # convert from float16 to float32
        dm = dm.float()
        dm = dm.mean(dim=2)
        # print(dm.dtype, dm.device)
        dm = interpolate(dm, (h, w), mode="bilinear")
        dm = dm[:, 0].unsqueeze(1) - dm[:, 1].unsqueeze(1)
        dm = -dm.squeeze(1).mean(dim=0).cpu().numpy()

        torch.cuda.empty_cache()
        return (path[0], np.mean(dm))
      except Exception as ex:
        print('error', ex)
        print('@path=', path)
        return None
   
    def filter_gt(path):
      # I apologize for my sins
      if gt_only:
        return path[1]
      else:
        return True

    paths = self.get_seeds(d, country)
    parallel = Parallel(n_jobs=n_jobs, timeout=100000)(delayed(compute)(path) for path in tqdm(paths, desc=f'Extracting D [{country}]'))
    return [p for p in parallel if p is not None]


  def extract_top_k_images(self, output_dir, k=5):
    for country in self.categories():
      os.makedirs(join(output_dir, 'ordered'), exist_ok=True)
      data = self.rank_images(country, gt_only=True)
      data_min = sorted(data, key=lambda x: x[1])
      data_max = sorted(data, key=lambda x: x[1], reverse=True)
      random.shuffle(data)

      # take five images of each and concat
      for name, data_ in zip(['D_least', 'D', 'random'], [data_min, data_max, data]):
        image = hcat_margin([self.resize(PIL.Image.open(path[0])).convert('RGBA') for path in data_[:k]])
        print('saving at', join(output_dir, 'ordered', country + '_' + name + '.png'))
        image.save(join(output_dir, 'ordered', country + '_' + name + '.png'))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--dataset_path', type=str, required=True)
  parser.add_argument('-c', '--cache_path', type=str, required=True)
  parser.add_argument('-t', '--typicality_path', type=str, required=True)
  parser.add_argument('-m', '--model_path', type=str, default=None)
  parser.add_argument('-w', '--which', type=str, required=True, choices=['ftt', 'geo', 'cars', 'places'])

  parser.add_argument('-s', '--seed', type=int, default=42)
  parser.add_argument('--recache', action='store_true')
  parser.add_argument('--not_cache_features', action='store_false')
  parser.add_argument('--topk', action='store_true')
  parser.add_argument('--figures_only', action='store_true')
  parser.add_argument('--top_full_images', action='store_true')
  parser.add_argument('--feature_which', type=str, default=None)
  parser.add_argument('--cluster', action='store_true')
  parser.add_argument('--umap', action='store_true')

  parser.add_argument('--max_row', type=int, default=6)
  parser.add_argument('--top_k_figure', type=int, default=5)
  parser.add_argument('--min_row', type=int, default=5)
  parser.add_argument('--grid_sep_x', type=int, default=2)
  parser.add_argument('--grid_sep_y', type=int, default=4)
  parser.add_argument('--aggregate', type=str, default='median', choices=['mean', 'median'])

  parser.add_argument('--figure_path', type=str, default=None)
  parser.add_argument('--num_images', type=int, default=None)
  parser.add_argument('--num_clusters', type=int, default=32)
  parser.add_argument('--k', type=int, default=64)
  args = parser.parse_args()

  cluster = Cluster(args.which, args.typicality_path, args.dataset_path, args.cache_path, args.recache, model_path=args.model_path, aggregate=args.aggregate, kx=args.k, ky=args.k, cache_features=args.not_cache_features)

  if not args.figures_only:
    if args.topk:
      cluster.plot_top_k(k_per_image=5, k=(50 if args.num_images is None else args.num_images))
    if args.cluster:
      cluster.clustering(k_per_image=5, k=(1000 if args.num_images is None else args.num_images), feature_which=args.feature_which, num_clusters=args.num_clusters, project=args.umap)

  if args.figure_path is not None:
    if args.top_full_images:
      cluster.extract_top_k_images(args.figure_path)
    else:
      cluster.make_figure(args.figure_path, topk=args.topk, feature_which=args.feature_which, hard_limit=args.max_row, top_k=args.top_k_figure, min_im=args.min_row, grid_sep_x=args.grid_sep_x, grid_sep_y=args.grid_sep_y)

