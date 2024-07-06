import time
import os
import math
import cv2
import torch
import shutil
import joblib
import clip
import random
import argparse
import numpy as np
from os.path import join
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
from itertools import chain
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from skimage.feature import hog
from joblib import Parallel, delayed
from utils import hcat, vcat, filter_by_contrast, iou, add_border
from hog import dense_search_cuda, get_hoglab_single, pre_safetensors, random_sample, normalize
from safetensors import safe_open
# from multiprocessing import Lock

import torchvision.transforms as transforms
import pickle
import numpy as np
import time

class Timer(object):
    def __init__(self, tag):
        self.tag = tag

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        print(f"{self.tag} took {minutes}m {seconds:.2f}s")

def accept_patch_neighbor(neighbors, output_detectors_init, buffers):
  buffer = defaultdict(list)
  for bbox, path in neighbors:
    buffer[path].append(bbox)

  for d in output_detectors_init:
    count = 0
    for path in buffers[d[0]]:
      bboxes = buffer[path]
      bboxesp = buffers[d[0]][path]
      # if between any combination of the bboxes ther is i.o.u. bigger than 0.3 set count += 1
      for bbox in bboxes:
        for bboxp in bboxesp:
          if iou(bbox + (bbox[0]+64, bbox[1]+64), bboxp + (bboxp[0]+64, bboxp[1]+64)) > 0.3:
            count += 1
            if count > 5:
              return buffer, False

  return buffer, True

def train_svm(X, split, max_samples, thunder=False):
  len_p, len_hn, len_n = split
  X = np.stack(X, axis=0)
  y = [1]*len_p + [-1]*(len_hn+len_n)
  # svm = LinearSVC(C=0.1, loss='hinge', dual=True)
  svm = SVC(C=0.1, kernel='linear')
  svm = svm.fit(X, y)

  # keep the top-max-samples hard negatives
  scores = svm.decision_function(X[len_p+len_hn:])
  idx = np.where(scores > 0)[0]
  sorted_idx = np.argsort(-scores[idx])  # Sort indices by the magnitude of decision function values
  hard_negatives = X[idx[sorted_idx][:max_samples] + len_p+len_hn]
  return svm.coef_[0], hard_negatives.tolist()

def search_batch(init_d_main_dir, ws, index, sft_paths, positive_paths_set, dev_i):
  locked = True
  device_id_fp = f'.device_{dev_i}'
  while locked:
    if os.path.isfile(device_id_fp):
      time.sleep(0.1)
    else:
      with open(device_id_fp, 'w') as f:
        f.write('locked')
        print(f'{device_id_fp} locked')
      locked = False

  buffer = dense_search_cuda(np.array(ws), sft_paths, top_k=50, device_id=f'cuda:{dev_i}')
  for i, (idx, bf) in enumerate(zip(index, buffer)):
    d20 = sum(map(lambda y: y[-1] in positive_paths_set, bf[:20]))
    neighbors = [(y[1], y[2]) for y in bf]

    fp = join(init_d_main_dir, f'{idx}.pkl')
    joblib.dump((ws[i], d20, neighbors), fp)
      
  torch.cuda.empty_cache()
  torch.cuda.synchronize()
  os.remove(device_id_fp)

# extract patches
class Doersch(object):
  def __init__(self, main_dir, which, seed=42, how_many=25000, threshold=50):
    self.which = which
    self.how_many = how_many
    self.main_dir = main_dir
    self.load_paths = (self.load_paths_geo if which == 'geo' else self.load_paths_ftt if which == 'ftt' else self.load_paths_cars)
    self.init_data_paths()
    self.load_paths(self.dataset_path)
    self.seed = 42
    self.threshold = threshold
    self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    self.hog_cache_path = f'{self.main_dir}/{self.which}/hog_cache'
    self.safetensors_path = f'{self.main_dir}/{self.which}/safetensors'
    self.preprocess()

  def flush_safetensors(self, c):
    shutil.rmtree(join(self.safetensors_path, f'{c}-all'), ignore_errors=True)
    shutil.rmtree(join(self.safetensors_path, f'{c}-neg'), ignore_errors=True)
    shutil.rmtree(join(self.safetensors_path, f'{c}-pos'), ignore_errors=True)
    if os.path.isfile(join(self.safetensors_path, f'{c}-all_paths.pkl')):
      os.remove(join(self.safetensors_path, f'{c}-all_paths.pkl'))
    if os.path.isfile(join(self.safetensors_path, f'{c}-neg_paths.pkl')):
      os.remove(join(self.safetensors_path, f'{c}-neg_paths.pkl'))
    if os.path.isfile(join(self.safetensors_path, f'{c}-pos_paths.pkl')):
      os.remove(join(self.safetensors_path, f'{c}-pos_paths.pkl'))

  def preprocess(self):
    # Compute HOG-LAB for all images and save to disk
    self.paths = defaultdict(list)
    for c in self.categories():
      for path in self.get_seeds(c):
        self.paths[c].append(path)

  def positive_paths(self, c, i=None, l=None):
    # get elements with random order seed should affect only this sort
    pos_path_file = f'{self.main_dir}/{self.which}/{c}/pos_all_{self.seed}_hog.pkl'
    if not os.path.isfile(pos_path_file):
      random.seed(self.seed)
      idexes = list(range(len(self.paths[c])))
      random.shuffle(idexes)
      os.makedirs(os.path.dirname(pos_path_file), exist_ok=True)
      joblib.dump([self.paths[c][i] for i in idexes], pos_path_file)

    paths = joblib.load(pos_path_file)
    if l is None:
      return paths
    else:
      return paths[len(paths)*i//self.l:len(paths)*(i+1)//self.l]

  def negative_paths(self, c, i=None, l=None):
    # get elements with random order seed should affect only this sort
    negative_path_file = f'{self.main_dir}/{self.which}/{c}/neg_all_{self.seed}_hog.pkl'
    if not os.path.isfile(negative_path_file):
      paths = []
      for cp, seed in zip(self.paths.keys(), range(self.seed*2, self.seed*2+len(self.paths))):
        random.seed(seed)
        idexes = list(range(len(self.paths[cp])))
        random.shuffle(idexes)
        if cp != c:
          paths += [self.paths[cp][i] for i in idexes]

      random.seed(self.seed*2+len(self.paths)+1)
      random.shuffle(paths)
      joblib.dump(paths, negative_path_file)

    paths = joblib.load(negative_path_file)
    if l is None:
      return paths
    else:
      return paths[len(paths)*i//self.l:len(paths)*(i+1)//self.l]

  def init_data_paths(self):
    if self.which == 'ftt':
        self.dataset_path = "/home/isig/diff-geo-mining/dataset/ftt/train"
    elif self.which == 'cars':
        self.dataset_path = "/home/isig/diff-geo-mining/dataset/cars/train"
    elif self.which == 'geo':
        self.dataset_path = "/home/isig/diff-geo-mining/doersch/geo-base-data"
    else:
        raise ValueError(f'Invalid dataset path: {self.which}')

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
      return list(self.parent.keys())
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

  def get_seeds(self, c):
    if self.which in {'ftt', 'cars'}:
      return [path for path in self.times[c]]
    elif self.which == 'geo':
      return [path[0] for path in self.country_path[c] if path[1]]

  def init_patches(self, c, how_many, num_trials=100):
    elems = {}
    for path in self.get_seeds(c):
      img = Image.open(path)
      elems[path] = np.zeros((img.size[0]//8 - 8, img.size[1]//8 - 8))

    pbar = tqdm(total=how_many, desc=f'Sampling {how_many} init patches')
    keys = list(elems.keys())
    key_id, total, patches = 0, 0, []

    random.shuffle(keys)
    while total < how_many:
      path = keys[key_id]
      X, Y = np.where(elems[path] == 0)
      idx = np.random.permutation(len(X))
      X, Y = X[idx][:num_trials], Y[idx][:num_trials]
      for j in range(len(X)):
        if len(elems[path]):
          elems[path][X[0], Y[1]] = 1
          bbox = (X[j]*8, Y[j]*8, X[j]*8 + 64, Y[j]*8 + 64)
          if filter_by_contrast(Image.open(path).crop(bbox), self.threshold):
            patches.append((bbox, path))
            pbar.update(1)
            total += 1
            break
        else:
          break
      key_id = (key_id+1)%len(keys)
    return patches

  def vis_patches(self, c, how_many, num_trials=100):
    elems = {}
    for path in self.get_seeds(c):
      img = Image.open(path)
      elems[path] = np.zeros((img.size[0]//8 - 8, img.size[1]//8 - 8))

    pbar = tqdm(total=how_many, desc=f'Sampling {how_many} init patches')
    keys = list(elems.keys())
    key_id, total, patches = 0, 0, []
    patches_con, patches_ncon = [], []

    random.shuffle(keys)
    while total < how_many:
      path = keys[key_id]
      X, Y = np.where(elems[path] == 0)
      idx = np.random.permutation(len(X))
      X, Y = X[idx][:num_trials], Y[idx][:num_trials]
      for j in range(len(X)):
        if len(elems[path]):
          elems[path][X[0], Y[1]] = 1
          bbox = (X[j]*8, Y[j]*8, X[j]*8 + 64, Y[j]*8 + 64)
          if filter_by_contrast(Image.open(path).crop(bbox), self.threshold):
            patches_con.append(Image.open(path).crop(bbox))
            pbar.update(1)
            total += 1
            break
          else:
            patches_ncon.append(Image.open(path).crop(bbox))
        else:
          break
      key_id = (key_id+1)%len(keys)
    hcat(patches_con[:100]).save(f'con-{c}.png')
    hcat(patches_ncon[:100]).save(f'ncon-{c}.png')
    return patches_con, patches_ncon

  @property
  def all_paths(self):
    return [path for category in self.categories() for path in self.paths[category]] 

  def init_detectors(self, c, patches, batch_size=64):
    positive_paths = self.positive_paths(c) 
    negative_paths = self.negative_paths(c)

    all_paths = positive_paths + negative_paths
    positive_paths_set = set(positive_paths)

    init_d_main_dir = join(self.main_dir, self.which, c, 'init')
    os.makedirs(init_d_main_dir, exist_ok=True)

    cache = []
    
    sft_paths = pre_safetensors(all_paths, self.hog_cache_path, self.safetensors_path, f'{c}-all')

    def gen_w():
      cache = []
      for index, (bbox, patch) in enumerate(patches):
        fp = join(init_d_main_dir, f'{index}.pkl')

        if not os.path.isfile(fp):
          w = normalize(np.load(join(self.hog_cache_path, os.path.abspath(patch).replace('/', '_')+'.npy')))[bbox[0]//8, bbox[1]//8]
          cache.append((index, w))

          if len(cache) == batch_size or (index == len(patches)-1 and len(cache)):
            ws = np.stack([w for _, w in cache], axis=0)
            index = [i for i, _ in cache]
            yield ws, index
            cache = []

    # get all available devices
    num_devices = torch.cuda.device_count()
    assert num_devices > 0
    # remove all paths if exist
    for i in range(num_devices):
      if os.path.isfile(f'.device_{i}'):
        os.remove(f'.device_{i}')

    total = math.ceil(sum(1 for index, (bbox, patch) in enumerate(patches) if not os.path.isfile(join(init_d_main_dir, f'{index}.pkl')))/batch_size)
    Parallel(n_jobs=num_devices, backend='loky')(delayed(search_batch)(init_d_main_dir, ws, index, sft_paths, positive_paths_set, k%num_devices) for k, (ws, index) in enumerate(tqdm(gen_w(), total=total, desc=f'Dense Search for Detector Init [{num_devices} gpus]')))

    metadata = {
      'discriminative-20': {},
      'neighbors': {},
      'w': {},
    }
    for index, (bbox, patch) in enumerate(patches):    
      fp = join(init_d_main_dir, f'{index}.pkl')
      w, d20, neighbors = joblib.load(fp)
      metadata['discriminative-20'][index] = d20
      metadata['neighbors'][index] = neighbors
      metadata['w'][index] = w

    return metadata

  def rank_init_detectors(self, num_detectors, c, stats, patches):
    output_detectors_init = []
    buffer = defaultdict(lambda: defaultdict(list))
    pbar = tqdm(desc='Ranking Detectors', total=num_detectors)
    for k, v in sorted(stats['discriminative-20'].items(), key=lambda x: x[1], reverse=True):
      if len(output_detectors_init) == num_detectors:
        break

      buffer_, flag = accept_patch_neighbor(stats['neighbors'][k], output_detectors_init, buffer)
      if flag:
        output_detectors_init.append((k, patches[k], stats['w'][k]))
        buffer[k] = buffer_
        pbar.update(1)

    return output_detectors_init

  def initialize_classifier(self, c, num_detectors=1000):
    index_init = f'{self.main_dir}/{self.which}/{c}/init_ws_{self.seed}_{self.threshold}_{self.how_many}_{num_detectors}_hog.pkl'
    if not os.path.isfile(index_init):
      init_patches_path = f'{self.main_dir}/{self.which}/{c}/init_patches_{self.seed}_{self.threshold}_{self.how_many}.pkl'
      if not os.path.isfile(init_patches_path):
        os.makedirs(os.path.dirname(init_patches_path), exist_ok=True)
        patches = self.init_patches(c, self.how_many)
        joblib.dump(patches, init_patches_path)
      patches = joblib.load(init_patches_path)
      stats = self.init_detectors(c, patches)
      init_points = self.rank_init_detectors(num_detectors, c, stats, patches)
      joblib.dump(init_points, index_init)

    return joblib.load(index_init)

  def make_detector_path(self, c, index, k_neighbors):
    path = f'{self.main_dir}/{self.which}/{c}/detectors/{self.threshold}/{k_neighbors}_{index}.pkl'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

  def pos_kfold(self, c, l):
    return [self.positive_paths(c, i, l) for i in range(l)]

  def neg_kfold(self, c, l):
    return [self.negative_paths(c, i, l) for i in range(l)]

  def safetensors_cache_path(self, c):
    return join(self.hog_cache_path,)

  def iterative_clustering(self, c, k=5, l=3, iterations=3, top_k=32, top_elem=7, num_detectors=1000, k_neighbors=5, batch_size=64, svm_batch_size=8, debug=True):
    self.c, self.l = c, l
    pos_paths_set = set(self.positive_paths(c))
    initial_indexes = self.initialize_classifier(c=c, num_detectors=num_detectors)

    all_paths = pre_safetensors(self.positive_paths(c) + self.negative_paths(c), self.hog_cache_path, self.safetensors_path, f'{c}-all')
    pos_fold = pre_safetensors(self.positive_paths(c), self.hog_cache_path, self.safetensors_path, f'{c}-pos', num_splits=1)
    neg_fold = pre_safetensors(self.negative_paths(c), self.hog_cache_path, self.safetensors_path, f'{c}-neg', num_splits=4)

    def gen_w():
      cache = []
      for index, (_, _, w) in enumerate(initial_indexes):
        if not os.path.isfile(self.make_detector_path(c, index, k_neighbors)):
          cache.append((index, w, self.make_detector_path(c, index, k_neighbors)))
  
        if len(cache) == batch_size or (index == len(initial_indexes)-1 and len(cache)):
          yield cache
          cache = []

    num_remaining = sum(1 for index, _ in enumerate(initial_indexes) if not os.path.isfile(self.make_detector_path(c, index, k_neighbors)))
    for cache in tqdm(gen_w(), total=math.ceil(num_remaining/batch_size), desc='Iterative Clustering'):
        ws = np.stack([w for _, w, _ in cache], axis=0)

        if debug:
          used_indices = [(index, initial_indexes[index]) for index, _, _ in cache]
          with Timer('search'):
            elem = dense_search_cuda(ws, all_paths, top_k=50)

          for e, c_ in zip(elem, cache):
            joblib.dump(e, c_[-1].replace('.pkl', f'_init.pkl'))

          self.plot_init(used_indices, c, k_neighbors, pos_paths_set) 

        hard_negatives = [[] for _ in range(len(cache))]
        for i in range(l):
          with Timer('search pos fold'):
            positives = dense_search_cuda(ws, pos_fold, fold=(i+1, l), top_k=5, ret_ws=True)

          if i == 0 and debug:
            for e, c_ in zip(positives, cache):
              joblib.dump(e, c_[-1].replace('.pkl', f'_init_fold.pkl'))
            self.plot_init_fold(used_indices, c, k_neighbors, pos_paths_set)

          with Timer('negative random sample'):
            negatives = [list(random_sample(neg_fold, fold=(i+1, l), num_samples=max(25000-len(hn), 10000))) for hn in tqdm(hard_negatives, desc=f'Negative Sampling [{i+1}/{l}]', disable=True)]

          def gen_batch():
            for p, n, hn in zip(positives, negatives, hard_negatives):
              yield [w for _, bbox, path, w in p] + hn + n, (len(p), len(hn), len(n))

          with Timer('training svms'):
            parallel = Parallel(n_jobs=svm_batch_size, timeout=10000000)(delayed(train_svm)(w, split, max(25000-split[1], 10000)) for w, split in tqdm(gen_batch(), total=len(positives), desc=f'Iterative Clustering [{i+1}/{l}]'))
            ws = []
            for idx, (w, negs) in enumerate(parallel):
              ws.append(w)
              hard_negatives[idx] += negs

          ws = np.stack(ws, axis=0)
          if debug:
            with Timer('training ws'):
              elem = dense_search_cuda(ws, all_paths, top_k=50)

            for e, c_ in zip(elem, cache):
              joblib.dump(e, c_[-1].replace('.pkl', f'_{i}.pkl'))
            self.plot_fold(used_indices, c, k_neighbors, i, pos_paths_set)

        with Timer('exit ws'):
          elem = dense_search_cuda(ws, all_paths, top_k=100)
          debug = False

        for i, (e, c_) in enumerate(zip(elem, cache)):
          accuracy = sum([1 for p in e if p[2] in pos_paths_set])
          top_detections = [(bbox, path) for _, bbox, path in e if path in pos_paths_set]
          joblib.dump((accuracy, e, top_detections, ws[i]), c_[-1])

    # self.plot_iterative(initial_indexes, c, l, k_neighbors)

    def load_elem(x):
      accuracy, top_detect, detector, _ = joblib.load(x)
      return accuracy, detector[:top_elem]

    data = [load_elem(self.make_detector_path(c, index, k_neighbors)) for index, _ in enumerate(initial_indexes)]
    return sorted(data, key=lambda x: x[0], reverse=True)[:top_k]

  def plot_init(self, available_indexes, c, k_neighbors, all_positive):
    base = f'{self.main_dir}/{self.which}/{c}/plots/{self.threshold}/detectors'
    os.makedirs(base, exist_ok=True)
    line = []
    for index, obj in available_indexes:
      _, (bbox, path), _ = obj
      paths = joblib.load(self.make_detector_path(c, index, k_neighbors).replace('.pkl', '_init.pkl'))
      line.append(
        hcat(
          [add_border(Image.open(path).crop(bbox), 2, 'white')] + 
          [add_border(Image.open(path_).crop((bbox_[0], bbox_[1], bbox_[0]+64, bbox_[1]+64)), 2, ('blue' if path_ in all_positive else 'red'))
           for k, (score, bbox_, path_) in enumerate(paths[:30])],
          margin=0
        )
      )
    vcat(line, margin=2).save(join(base, f'init.png'))

  def plot_init_fold(self, available_indexes, c, k_neighbors, all_positive):
    base = f'{self.main_dir}/{self.which}/{c}/plots/{self.threshold}/detectors'
    os.makedirs(base, exist_ok=True)
    line = []
    for index, obj in available_indexes:
      _, (bbox, path), _ = obj
      paths = joblib.load(self.make_detector_path(c, index, k_neighbors).replace('.pkl', '_init_fold.pkl'))
      line.append(
        hcat(
          [add_border(Image.open(path).crop(bbox), 2, 'white')] + 
          [add_border(Image.open(path_).crop((bbox_[0], bbox_[1], bbox_[0]+64, bbox_[1]+64)), 2, ('blue' if path_ in all_positive else 'red'))
           for k, (score, bbox_, path_, _) in enumerate(paths[:30])],
          margin=0
        )
      )
    vcat(line, margin=2).save(join(base, f'init_fold.png'))

  def plot_fold(self, available_indexes, c, k_neighbors, i, all_positive):
    base = f'{self.main_dir}/{self.which}/{c}/plots/{self.threshold}/detectors'
    os.makedirs(base, exist_ok=True)
    line = []
    for index, _ in available_indexes:
      paths = joblib.load(self.make_detector_path(c, index, k_neighbors).replace('.pkl', f'_{i}.pkl'))
      line.append(
        hcat(
          [add_border(Image.open(path_).crop((bbox_[0], bbox_[1], bbox_[0]+64, bbox_[1]+64)), 2, ('blue' if path_ in all_positive else 'red'))
            for k, (score, bbox_, path_) in enumerate(paths[:30])],
          margin=0
        )
      )

    img = vcat(line, margin=2)
    img.save(join(base, f'{i}.png'))

  def plot_iterative(self, indexes, c, l, k_neighbors):
    all_positive = set(self.positive_paths(c))
    available_indexes = []
    for index, obj in enumerate(indexes):
      elems = [self.make_detector_path(c, index, k_neighbors).replace('.pkl', f'_init.pkl')]
      elems += [self.make_detector_path(c, index, k_neighbors).replace('.pkl', f'_{i}.pkl') for i in range(l)]
      if all(os.path.isfile(elem) for elem in elems):
        available_indexes.append((index, obj))

      if len(available_indexes) == 30:
        break

    assert len(available_indexes)

    self.plot_init(available_indexes, c, k_neighbors, all_positive)
    self.plot_init_fold(available_indexes, c, k_neighbors, all_positive)

    for i in range(l):
      self.plot_fold(available_indexes, c, k_neighbors, i, all_positive)

  def get_top(self, c, top_k=32, top_elem=7):
    data = self.iterative_clustering(c=c, top_k=top_k, top_elem=top_elem)
    fname = f'{self.main_dir}/{self.which}/{c}/top_{self.seed}_{self.threshold}_{self.how_many}_hog_final.png'
    line = []
    for acc, indexes in data:
      line.append(hcat([Image.open(path).crop((bbox[0], bbox[1], bbox[0]+64, bbox[1]+64)) for bbox, path in indexes], margin=2))
    img = vcat(line, margin=4)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    img.save(fname)
    return img


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Doersch')
  parser.add_argument('--threshold', type=int, default=50, help='Threshold')
  parser.add_argument('--how_many', type=int, default=25000, help='Threshold')
  parser.add_argument('--main_dir', type=str, default='doersch-hog', help='Main directory')
  parser.add_argument('--which', type=str, default='geo', choices=['ftt', 'cars', 'geo'], help='Which')
  parser.add_argument('--category', type=str, default='United States', help='Category')
  args = parser.parse_args()


  doersch = Doersch(main_dir=args.main_dir, which=args.which, how_many=args.how_many, threshold=args.threshold)
  # categories = list(Doersch(main_dir=args.main_dir, which=args.which, how_many=args.how_many, threshold=args.threshold).categories())
  # for category in ['United States', 'Russia', 'Thailand', 'Brazil', 'France', 'Japan', 'Italy', 'Nigeria', 'India'][1:]:
  category = args.category
  doersch = Doersch(main_dir=args.main_dir, which=args.which, how_many=args.how_many, threshold=args.threshold)
  # doersch.vis_patches(category, 100)
  doersch.get_top(c=category)

  # cleanup
  doersch.flush_safetensors(category)
  # del doersch
  # torch.cuda.empty_cache()
  # torch.cuda.synchronize()
  # import gc
  # gc.collect()
