import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

import operator
import random
import copy
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
from sklearn.cluster import KMeans

import torch
import umap
from torch.nn.functional import interpolate
from diffmining.typicality.utils import sort, pool, get_top_k, get_non_overlapping, hcat_rgba_border, add_border, normalize, filter_patch, hcat, hcat_margin, apply_alpha, make_grid, vcat
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
  def __init__(self, typicality_path, dataset_path, cache_path, recache, model_path, aggregate='median', kx=64, ky=64, xformers=False):
    super().__init__(model_path=None, dataset_path=dataset_path, typicality_path=typicality_path, xformers=xformers)
    self.cache_path = cache_path
    self.recache = recache
    self.kx = kx
    self.ky = ky
    self.countries = ['United States', 'Japan', 'France', 'Italy', 'United Kingdom', 'Brazil', 'Russia', 'Thailand', 'Nigeria', 'India']
    self.model_path = model_path
    self.aggregate = (mean, median)[int(aggregate == 'median')]

  def get_seeds(self, d, tag):
    return [path[0] for path in self.country_path[tag] if d.exists(path[0])]

  def load_image(self, path):
    img = PIL.Image.open(path).convert('RGB')
    img = self.resize(img)
    return img

  def load_and_apply_alpha_bbox(self, country, pil, idd, e):
    # get the bbox of pild
    bbox, country, path = self.decompose_save_path(idd, e, country)
    alpha = self.load_individual_(path)

    # crop alpha to bbox
    I = np.array(self.load_image(path))/255.0
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

  def load_typicallity_norm(self, d, path):
    w, h = self.load_image(path).size
    dm = d(path)
    dm = torch.from_numpy(dm).to(self.device)
    
    # convert from float16 to float32
    dm = dm.float()
    dm = dm.mean(dim=2)
    
    dm = interpolate(dm, (h, w), mode="bilinear")
    dm = (dm[:, 1] - dm[:, 0]).mean(dim=0).cpu().numpy()
    dm = normalize(dm)
    return dm

  @torch.no_grad()
  def load_typicallity(self, d, path):
    w, h = self.load_image(path).size
    dm = d(path)
    dm = torch.from_numpy(dm).to(self.device)
    
    # convert from float16 to float32
    dm = dm.float()
    dm = dm.mean(dim=2)

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
    return pil

  def init__clip(self, clip_model="models/clip-vit-base-patch32"):
    if not hasattr(self, 'clip'):
      from transformers import CLIPProcessor, CLIPModel
      self.processor = CLIPProcessor.from_pretrained(clip_model)
      self.clip_model = CLIPModel.from_pretrained(clip_model).to(self.device)

  @torch.no_grad()
  def embed_batch(self, images, clip, dift, t, id, bbox):
    if clip:
      pkl_file = join(self.cache_path, 'embeddings', 'clip', f'{id}.pkl')
      if not os.path.isfile(pkl_file):
        image_input = self.processor(images=[img.crop(bbox) for img in images], return_tensors="pt")
        features = self.clip_model.get_image_features(**image_input.to(self.device))
        features /= features.norm(dim=-1, keepdim=True)
        clip_features = features.flatten().cpu().numpy()
        joblib.dump(clip_features, pkl_file)
      else:
        clip_features = joblib.load(pkl_file)

    if dift:
      pkl_file = join(self.cache_path, 'embeddings', f'dift-{t}', f'{id}.pkl')
      if not os.path.isfile(pkl_file):
        # in this case we put red if element is ground truth and transparent otherwise
        y_start, x_start, y_end, x_end = bbox
        dift_features = []
        for c, pil in zip(self.countries, images):
          dift_features_ = self.dift.forward(dift_pre(pil), t=t, prompt=self.dift_prompt(c), ensemble_size=8).squeeze(0).cpu().numpy()
          c, h, w = dift_features_.shape
          H = h/pil.height
          W = w/pil.width

          dift_features_ = dift_features_[:, int(x_start*H):int(x_end*H), int(y_start*W):int(y_end*W)].mean(axis=(1, 2))
          dift_features.append(dift_features_ / np.linalg.norm(dift_features_))

        dift_features = np.concatenate(dift_features)
        joblib.dump(dift_features, pkl_file)
      else:
        dift_features = joblib.load(pkl_file)

    if clip and dift:
      return np.concatenate([clip_features, dift_features])
    elif clip:
      return clip_features
    else:
      return dift_features
      
  def dift_prompt(self, c):
    return f"{c}" if len(c) else ""

  def compute_embeddings(self, df, feature_which='dift-261'):
    X, ids, pils, ds, orig_path = [], [], [], [], []
    
    dift = 'dift' in feature_which
    clip = 'clip' in feature_which
    if clip:
        self.init__clip()

    if clip and dift:
      t = int(feature_which.split('+')[1].split('-')[1])
    elif dift:
      t = int(feature_which.split('-')[1])

    for i in trange(df.shape[0]):
      ds.append(df.iloc[i]['D'])
      orig_path.append(df.iloc[i]['origin'])
      pilito = [PIL.Image.open(df.iloc[i]['path_' + c]) for c in self.countries]

      path__ = os.path.split(df.iloc[i]['path_' + df.iloc[i]['origin']])[1]
      x_start, y_start, x_end, y_end = df.iloc[i]['x_start'], df.iloc[i]['y_start'], df.iloc[i]['x_end'], df.iloc[i]['y_end']
      ids.append(path__.replace(os.path.splitext(path__)[1], '_') + f'{x_start}-{y_start}-{x_end}-{y_end}')

      # can you compute this in batch?
      X.append(self.embed_batch(pilito, clip, dift, t, ids[-1], (y_start, x_start, y_end, x_end)))
      pilito = [add_border(pilito[j].crop((y_start, x_start, y_end, x_end)), 'transparent' if c != df.iloc[i]['origin'] else 'red')  for j, c in enumerate(self.countries)]
      pils.append(hcat(pilito))

    return X, ids, pils, ds, orig_path

  @torch.no_grad()
  def df_PD(self, k_per_image, seed=42, n_jobs=12, ascending=False):
    column = ['x_start', 'y_start', 'x_end', 'y_end', 'origin', 'D'] + self.countries + ['path_' + c for c in self.countries]
    def compute(country_origin, paths):
      try:
        pths = {country: path for path, country in paths}
        ds = {country: self.load_typicallity(self.D[country], path) for path, country in paths}
        dm = np.median(np.stack([ds[c] for c in self.countries], axis=0), axis=0)

        df = [(i, j, i+self.kx, j+self.ky, country_origin, dm[i, j]) + tuple([ds[c][i, j] for c in self.countries]) + tuple([pths[c] for c in self.countries]) for i in range(dm.shape[0]) for j in range(dm.shape[1])]
        df_random = copy.deepcopy(df)
        random.shuffle(df_random)
        df_random = pd.DataFrame(df_random, columns=column)

        df = pd.DataFrame(df, columns=column)
        df = sort(df, 'D', ascending=ascending)

        return get_non_overlapping(df, k_per_image=k_per_image), get_non_overlapping(df_random, k_per_image=k_per_image)
      except Exception as ex:
        import traceback
        print(traceback.format_exc())
        print('@path=', paths)
        return pd.DataFrame([], columns=column)

    paths = [(c, path) for c in self.countries for path in self.parallel[c] if all(self.D[country].exists(path_) for path_, country in path)]
    parallel = Parallel(n_jobs=n_jobs)(delayed(compute)(c, paths) for c, paths in tqdm(paths, desc=f'Extracting PD'))
    topk, randomized = zip(*parallel)
    return pd.concat([df for df in topk], axis=0), pd.concat([df for df in randomized], axis=0)

  def compress(self, X, num_components=32, n_neighbors=15):
    emb_size = len(X[0])
    group_size = emb_size // len(self.countries)

    embeddings_reduced = []
    for i in range(0, emb_size, group_size):
        embeddings_group = [x[i:i+group_size] for x in X]
        reducer = umap.UMAP(n_components=num_components, n_neighbors=n_neighbors)
        embeddings_group_reduced = reducer.fit_transform(embeddings_group)
        if len(embeddings_reduced) == 0:
            embeddings_reduced = embeddings_group_reduced
        else:
            embeddings_reduced = np.hstack((embeddings_reduced, embeddings_group_reduced))
    return embeddings_reduced

  def cluster(self, X, ids, pils, ds, real_paths, num_clusters=32, num_components=32, n_neighbors=15):
    clustering = KMeans(n_clusters=num_clusters, random_state=10)
    X_reduced = self.compress(X, num_components=num_components, n_neighbors=n_neighbors)

    clustering.fit(X_reduced)
    clusters = defaultdict(list)
    for pil, (i, l) in zip(pils, enumerate(clustering.labels_)):
      clusters[l].append((pil, ds[i], ids[i], X[i], real_paths[i]))

    centers = []
    for cluster_center in clustering.cluster_centers_:
        distances = np.linalg.norm(X_reduced - cluster_center[None, :], axis=1)
        closest_index = np.argmax(distances)
        centers.append(X[closest_index])

    # find the centroid of each cluster
    for k, vs in clusters.items():
      # find the point that is closer to the cluster center
      clusters[k] = [(a, b, c, e) for a, b, c, d, e in sorted(vs, key=lambda x: np.linalg.norm(x[-2] - centers[k]))]

    # Assuming 'X' is the data and 'clustering.labels_' are the cluster labels
    return sorted([(vs, self.aggregate(vs), ) for _, vs in clusters.items()], key=operator.itemgetter(1), reverse=True)

  def clustering(self, html_path, feature_which, k_per_image=5, k=1000, num_clusters=32, num_components=32, only_gt=True):
    def copy_to_images_dir(pil, parent_, tag, local_dir):
      os.makedirs(parent_, exist_ok=True)
      if not os.path.isfile(tag):
        pil.save(join(parent_, tag))
      return join(local_dir, tag)

    cache_path = join(self.cache_path, 'clusters')
    fp = os.path.join(cache_path, 'all.pkl')
    if not os.path.isfile(fp) or self.recache:
      os.makedirs(cache_path, exist_ok=True)
      df, df_random = self.df_PD(k_per_image=k_per_image)
      joblib.dump((df, df_random), fp)

    df, df_random = joblib.load(os.path.join(cache_path, 'all.pkl'))
    df = get_top_k(df, key='D', k=k, randomize=False)
    df_random = get_top_k(df_random, key='D', k=k, randomize=True)
    if 'dift' in feature_which:
      self.dift = SDFeaturizer(self.model_path, text_encoder_id="geolocal/StreetCLIP")

    os.makedirs(join(self.cache_path, 'embeddings', feature_which), exist_ok=True)
    for randomize in [False]:
      random_tag = ('ranked', 'random')[int(randomize)]
      embs = self.compute_embeddings((df_random if randomize else df), feature_which=feature_which)
      clusters = self.cluster(*embs, num_clusters=num_clusters, num_components=num_components)
      parent_ = join(self.cache_path, 'images', f'clusters', str(k), str(num_clusters), random_tag, feature_which)
      os.makedirs(parent_, exist_ok=True)

      # column should be the cluster id
      for i in range(num_clusters):
        for j, (pil, ds, idd, _) in enumerate(clusters[i][0]):
          pil.save(join(parent_, f'{i}-{j}-{num_clusters}_{idd}.png'))


  def make_figure(self, html_path, figure_path, k, num_clusters, hard_limit=6, top_k=5, min_im=5, feature_which=None, topk=False):
    # assumes that html has been already created
    # assumes that the top-k images have been already extracted
    for which in ['ranked']: #, 'random']:
      dirr = join(html_path, 'images', f'clusters', str(k), str(num_clusters), which)

      if '/ft' in figure_path and which == 'random':
        continue
      if not os.path.isdir(dirr):
        continue
      for feature_type in os.listdir(dirr):
        if feature_which != 'all' and feature_type != feature_which:
          continue

        group = defaultdict(lambda : defaultdict(list))
        for image_path in os.listdir(join(dirr, feature_type)):
          image_path_ = join(dirr, feature_type, image_path)
          cluster_id, idx, num_clusters = image_path.split('-')[:3]
          num_clusters = num_clusters.split('_')[0]
          group[num_clusters][cluster_id].append((idx, image_path_))

        # collect the top-5 predictions that don't have less than 6 elements and put them in a grid with a spacing of 2 pixels per column and 4 pixels per line
        for nc, b in group.items():
          parent = join(figure_path, 'clusters', which, feature_type, nc)
          os.makedirs(parent, exist_ok=True)

          images_grid = []
          bs = list(b.items())
          for i, cluster_elems in sorted(bs, key=lambda x: int(x[0])):
            images = sorted(cluster_elems, key=lambda x: int(x[0]))
            if len(images_grid) == top_k:
              break
            if len(images) < min_im:
              continue

            vcat([PIL.Image.open(image_path_).convert('RGB') for _, image_path_ in images[:hard_limit]], vertical_spacing=1).save(join(parent, f'{i}__hard_limit_{hard_limit}__top_k_{top_k}__min_im_{min_im}.png'))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--dataset_path', type=str, required=True)
  parser.add_argument('-t', '--typicality_path', type=str, required=True)
  parser.add_argument('-c', '--cache_path', type=str, required=True)
  parser.add_argument('-m', '--model_path', type=str, default=None)

  parser.add_argument('-s', '--seed', type=int, default=42)
  parser.add_argument('--recache', action='store_true')
  parser.add_argument('--cache_features', action='store_true')
  parser.add_argument('--cluster', action='store_true')
  parser.add_argument('--topk', action='store_true')
  parser.add_argument('--figures_only', action='store_true')
  parser.add_argument('--top_full_images', action='store_true')
  parser.add_argument('--feature_which', type=str, default=None)

  parser.add_argument('--max_row', type=int, default=6)
  parser.add_argument('--top_k_figure', type=int, default=5)
  parser.add_argument('--min_row', type=int, default=5)
  parser.add_argument('--aggregate', type=str, default='median', choices=['mean', 'median'])

  parser.add_argument('--figure_path', type=str, default=None)
  parser.add_argument('--num_images', type=int, default=None)
  parser.add_argument('--num_clusters', type=int, default=32)
  parser.add_argument('--num_components', type=int, default=32)
  parser.add_argument('--k', type=int, default=64)
  args = parser.parse_args()

  cluster = Cluster(args.typicality_path, args.dataset_path, args.cache_path, args.recache, model_path=args.model_path, aggregate=args.aggregate, kx=args.k, ky=args.k)
  k = (10000 if args.num_images is None else args.num_images)
  num_clusters = args.num_clusters
  if not args.figures_only:
    if args.topk:
      cluster.plot_top_k(k_per_image=5, k=(50 if args.num_images is None else args.num_images))
    if args.cluster:
      cluster.clustering(args.cache_path, k_per_image=5, k=k, feature_which=args.feature_which, num_clusters=num_clusters, num_components=args.num_components)

  if args.figure_path is not None:
    if args.top_full_images:
      cluster.extract_top_k_images(args.figure_path)
    else:
      cluster.make_figure(args.cache_path, args.figure_path, k=k, num_clusters=num_clusters, topk=args.topk, feature_which=args.feature_which, hard_limit=args.max_row, top_k=args.top_k_figure, min_im=args.min_row)

