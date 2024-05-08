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

# def get_top_k(df, k_per_image=5, k=1000, key='V', randomize=False, ascending=False):
#   k = min(len(df), k)
#   if randomize:
#     return df.sample(k)
#   else:
#     df = sort(df, key=key, ascending=ascending)
#     return df.iloc[:k]


def get_non_overlapping(df, k_per_image=5, merge_close_boxes=False):
  non_overlapping = []
  while len(non_overlapping) < k_per_image:
    non_overlapping.append(df.iloc[0])
    df = df[~((df['x_start'] <= non_overlapping[-1]['x_end']) & (df['x_end'] >= non_overlapping[-1]['x_start']) & (df['y_start'] <= non_overlapping[-1]['y_end']) & (df['y_end'] >= non_overlapping[-1]['y_start']))].reset_index(drop=True)
    if df.shape[0] == 0:
      break

  return pd.DataFrame(non_overlapping, columns=df.columns)

def filter_patch(pil, black_threshold=30, white_threshold=225):
  # Convert the image to grayscale and then to a numpy array
  grayscale = pil.copy().convert('L')
  np_image = np.array(grayscale)
  mean_pixel_value = np.mean(np_image)
  return black_threshold < mean_pixel_value < white_threshold

def hcat_margin(pils, margin=2):
  widths, heights = zip(*(i.size for i in pils))
  total_width = sum(widths) + margin*(len(pils) - 1)
  max_height = max(heights)
  new_im = PIL.Image.new('RGB', (total_width, max_height))
  x_offset = 0
  for im in pils:
    new_im.paste(im, (x_offset, 0))
    x_offset += im.size[0] + margin
  return new_im

def d_compute(dm, h, w, x_start, y_start, x_end, y_end):
  dm = torch.from_numpy(dm)

  # convert from float16 to float32
  dm = dm.float()
  dm = dm.mean(dim=2)

  dm = interpolate(dm, (h, w), mode="bilinear")
  dm = (dm[:, 1] - dm[:, 0]).mean(dim=0).cpu().numpy()

  dm = dm/np.max(np.abs(dm))
  dm = dm[x_start:x_end, y_start:y_end]
  return dm

def load_typicallity_norm_cars(time, image_id, tag, x_start, y_start, x_end, y_end):
  CACHE = '/home/isig/diff-geo-mining/gold/typicality/cars/'
  DATASET = '/home/isig/diff-geo-mining/dataset/cars/train'
  path = join(DATASET, image_id + '.jpg')

  w, h = PIL.Image.open(path).size
  if w > h:
    w = int(w * 256 / h)
    h = 256
  else:
    h = int(h * 256 / w)
    w = 256

  dm = np.load(join(CACHE, tag, time, image_id + '.npy'))
  return d_compute(dm, h, w, x_start, y_start, x_end, y_end)

def load_typicallity_norm_faces(time, image_id, tag, x_start, y_start, x_end, y_end):
  CACHE = '/home/isig/diff-geo-mining/gold/typicality/faces/'
  dm = np.load(join(CACHE, tag, time, image_id + f'_{time}.png'))
  return d_compute(dm, 256, 256, x_start, y_start, x_end, y_end)

def load_typicallity_norm_geo(country, image_id, tag, x_start, y_start, x_end, y_end):
  CACHE = '/home/isig/diff-geo-mining/gold/typicality/geo/'
  DATASET = '/home/isig/diff-geo-mining/dataset/parallel-2'

  w, h = PIL.Image.open(join(DATASET, country, 'gt--' + image_id + '.jpg')).size
  dm = np.load(join(CACHE, tag, country, 'gt--' + image_id + '.npy'))
  return d_compute(dm, h, w, x_start, y_start, x_end, y_end)

def apply_alpha(pil_path, sigma=10, recompute=False):
  parent, path = os.path.split(pil_path)
  ext = os.path.splitext(path)[1]
  path_alpha = join(parent, 'alpha-' + path.replace(ext, '.pkl'))
  ext = os.path.splitext(path)[1]
  if recompute:
    tag = ('ft' if 'ft/' in pil_path else 'pt')
    if 'geo/' in pil_path:
      path_parts = pil_path.split('__')
      country = '--'.join(pil_path.split('--')[1:]).split('__')[0]
      image_id = '_'.join('--'.join(pil_path.split('--')[1:]).split('_')[:-1])
      ext = os.path.splitext(pil_path)[1]
      x_start, y_start, x_end, y_end = [int(coord) for coord in '--'.join(os.path.split(pil_path)[1].split('--')[1:]).split('_')[-1].replace(ext, '').split('-')[0:4]]
      image_T_array = load_typicallity_norm_geo(country, image_id, tag, x_start, y_start, x_end, y_end)
    elif 'ftt/' in pil_path:
      image_id = os.path.split(pil_path)[1].split('_')[1]
      year = os.path.split(pil_path)[1].split('_')[2]
      ext = os.path.splitext(pil_path)[1]
      x_start, y_start, x_end, y_end = [int(coord) for coord in os.path.split(pil_path)[1].split('_')[-1].replace(ext, '').split('-')[0:4]]
      image_T_array = load_typicallity_norm_faces(year, image_id, tag, x_start, y_start, x_end, y_end)
    elif 'cars/' in pil_path:
      parent, path = os.path.split(pil_path)
      year = os.path.split(parent)[1]
      image_id = path.split('_')[1]
      ext = os.path.splitext(path)[1]
      x_start, y_start, x_end, y_end = [int(coord) for coord in path.split('_')[-1].replace(ext, '').split('-')]
      image_T_array = load_typicallity_norm_cars(year, image_id, tag, x_start, y_start, x_end, y_end)
    else:
      raise ValueError('Unknown category')

    joblib.dump(image_T_array, path_alpha)
  else:
    try:
      image_T_array = joblib.load(path_alpha)
    except:
      return apply_alpha(pil_path, sigma=sigma, recompute=True)

  pil = PIL.Image.open(pil_path).convert('RGB') 
  image_array = np.array(pil)/255.0

  filtered_image_t = gaussian_filter(image_T_array, sigma=sigma)
  filtered_image_t = filtered_image_t/np.max(filtered_image_t)
  filtered_image_t = filtered_image_t*(filtered_image_t>0)

  I = image_array
  T = filtered_image_t
  T = np.stack((T,T,T), axis=-1)

  R = 0.05*I + 0.95*(T*I + (1-T))
  return PIL.Image.fromarray((R*255).astype(np.uint8))

import PIL
import pandas as pd
from skimage import exposure
from skimage import filters

def load_image(x, resize_func=lambda x: x):
  pil = resize_func(PIL.Image.open(x['seed']))/255.0
  if 'x_start' not in x.columns:
    x_start, y_start, x_end, y_end = 0, 0, pil.size[1], pil.size[0]
  else:
    x_start, y_start, x_end, y_end = x[['x_start', 'y_start', 'x_end', 'y_end']]

  return pil.crop((y_start, x_start, y_end, x_end))

def filter_by_contrast(x, fraction_threshold=0.05, lower_percentile=1, upper_percentile=99, method='linear'):
  return not exposure.is_low_contrast(x, fraction_threshold=fraction_threshold, lower_percentile=lower_percentile, upper_percentile=upper_percentile, method=method)

def filter_by_gradient(x, fraction_threshold=0.05, lower_percentile=0.01, upper_percentile=0.99):
  footprint = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
  return np.mean(filters.rank.gradient_percentile(x, footprint, out=None, mask=None, shift_x=False, shift_y=False, p0=lower_percentile, p1=upper_percentile)) > fraction_threshold

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

from sklearn.metrics.pairwise import _euclidean_distances_upcast
def _euclidean_distances(X, Y, Y_norm_squared=None):
    XX = None
    YY = Y_norm_squared.reshape(1, -1)
    distances = _euclidean_distances_upcast(X, XX, Y, YY)
    np.maximum(distances, 0, out=distances)
    return distances

def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    expected = np.sum(arr, axis=axis, dtype=np.float64)
    if not np.allclose(
        out.take(-1, axis=axis), expected, rtol=rtol, atol=atol, equal_nan=True
    ):
        import warnings
        warnings.warn(
            (
                "cumsum was found to be unstable: "
                "its last element does not correspond to sum"
            ),
            RuntimeWarning,
        )
    return out

def kmeanspp_init(
    X, n_clusters, x_squared_norms, sample_weight, n_local_trials=None
):
    n_samples, n_features = X.shape
    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly and track index of point
    center_id = np.random.choice(n_samples, p=sample_weight / sample_weight.sum())
    indices = np.full(n_clusters, -1, dtype=int)
    centers[0] = X[center_id]
    indices[0] = center_id

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = _euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms
    )
    current_pot = closest_dist_sq @ sample_weight

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = np.random.uniform(size=n_local_trials) * current_pot
        candidate_ids = np.searchsorted(
            stable_cumsum(sample_weight * closest_dist_sq), rand_vals
        )

        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = _euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms
        )

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates @ sample_weight.reshape(-1, 1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        centers[c] = X[best_candidate]
        indices[c] = best_candidate

    return centers, indices



# class KMeansBase(object):
#   def __init__(self, n_clusters, init='kmeans++', max_iters=100, random_state=32):
#     self.n_clusters = n_clusters
#     self.max_iters = max_iters
#     self.random_state = random_state
#     self.init = init
#     np.random.seed(random_state)

#   def fit(self, X):
#     # Initialize centroids randomly
#     if self.init == 'random':
#       self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
#     else:
#       self.centroids = kmeanspp_init(X, self.n_clusters, np.ones(X.shape[0]), np.ones(X.shape[0]))[0]      
    
#     for _ in range(self.max_iters):
#       # Assign each data point to the nearest centroid
#       labels = self._assign_labels(X)
      
#       # Update centroids
#       new_centroids = self._update_centroids(X, labels)
      
#       # Check for convergence
#       if np.all(self.centroids == new_centroids):
#         break

#       self.centroids = new_centroids

#     labels = self._assign_labels(X)
#     self.labels_ = labels
#     self.cluster_centers_ = self.centroids
#     return self

#   def _assign_labels(self, X):
#       # Compute distances from each data point to centroids
#       distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

#       # Assign labels based on the nearest centroid
#       return np.argmin(distances, axis=1)
  
#   def _update_centroids(self, X, labels):
#       new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
#       return new_centroids

from sklearn.utils.extmath import row_norms
from sklearn.cluster._kmeans import _kmeans_single_lloyd, _kmeans_plusplus, _is_same_clustering

class KMeansBase(object):
  def __init__(self, n_clusters, init='kmeans++', max_iters=300, tolerance=0, random_state=32, n_init=10):
    self.n_clusters = n_clusters
    self.max_iters = max_iters
    self.random_state = random_state
    self.init = init
    self.tol = tolerance
    self.n_init = n_init
    #init random state
    self.random_state = np.random.RandomState(random_state)

  def fit(self, X):
    # Initialize centroids randomly
    Xmean = np.mean(X, axis=0)
    X = X - Xmean
    X = X.astype(np.float64)

    best_inertia, best_labels, best_centers, best_n_iter = None, None, None, None
    for _ in range(self.n_init):
        if self.init == 'random':
            centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        elif self.init == 'kmeans++':
            x_squared_norms = row_norms(X, squared=True)
            centroids = _kmeans_plusplus(X, self.n_clusters, x_squared_norms=x_squared_norms, random_state=self.random_state)[0]

        labels, inertia, centroids, n_iter_ = _kmeans_single_lloyd(
            X.astype(np.float64),
            np.ones(X.shape[0], dtype=np.float64),
            centroids.astype(np.float64),
            max_iter=self.max_iters,
            verbose=False,
            tol=self.tol,
            n_threads=1,
        )

        if best_inertia is None or (
            inertia < best_inertia
            and not _is_same_clustering(labels, best_labels, self.n_clusters)
        ):
            best_labels = labels
            best_centers = centroids
            best_inertia = inertia
            best_n_iter = n_iter_

    self.labels_ = best_labels
    self.cluster_centers_ = best_centers + Xmean
    return self

class KMeansRe(object):
  def __init__(self, n_clusters, init='kmeans++', max_iters=300, tolerance=0, random_state=32, n_init=10, k_min=0.01):
    self.n_clusters = n_clusters
    self.max_iters = max_iters
    self.random_state = random_state
    self.init = init
    self.tol = tolerance
    self.n_init = n_init
    #init random state
    self.k_min = k_min
    self.random_state = np.random.RandomState(random_state)

  def fit(self, X):
    # Initialize centroids randomly
    Xmean = np.mean(X, axis=0)
    X = X - Xmean
    X = X.astype(np.float64)

    best_inertia, best_labels, best_centers, best_n_iter = None, None, None, None
    for _ in range(self.n_init):
        if self.init == 'random':
            centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        elif self.init == 'kmeans++':
            x_squared_norms = row_norms(X, squared=True)
            centroids = _kmeans_plusplus(X, self.n_clusters, x_squared_norms=x_squared_norms, random_state=self.random_state)[0]

        change = True
        while change:
          labels, inertia, new_centroids, n_iter_ = _kmeans_single_lloyd(
              X.astype(np.float64),
              np.ones(X.shape[0], dtype=np.float64),
              centroids.astype(np.float64),
              max_iter=self.max_iters,
              verbose=False,
              tol=self.tol,
              n_threads=1,
          )

          # Check for convergence
          if np.all(centroids == new_centroids):
            break

          change, centroids = self.split_reassign(np.copy(new_centroids), labels, X)

        if best_inertia is None or (
            inertia < best_inertia
            and not _is_same_clustering(labels, best_labels, self.n_clusters)
        ):
            best_labels = labels
            best_centers = new_centroids
            best_inertia = inertia
            best_n_iter = n_iter_            

    self.labels_ = best_labels
    self.cluster_centers_ = best_centers + Xmean
    return self

  def split_reassign(self, centroids, labels, X):
    change = False
    remove_ids = []
    for i in range(self.n_clusters):
      if (labels == i).sum() < self.k_min*X.shape[0]:
        remove_ids.append(i)

    if len(remove_ids) > self.n_clusters//2:
      import warnings; warnings.warn(f'{len(remove_ids)}, {self.n_clusters}: too many splits skipping')
      return False
    
    if len(remove_ids) > 0:
      change = True
      # assign all centroids to the cluster with the most elements
      most_elements = -1
      for i in range(self.n_clusters):
        if (labels == i).sum() > most_elements:
          most_elements = (labels == i).sum()
          most_elements_id = i

      # compute the sigma of all elements in the max centroid
      sigma = np.std(X[labels == most_elements_id], axis=0)
      for i in remove_ids:
        centroids[i] = centroids[most_elements_id] + np.random.normal(0, 0.01*sigma, centroids[most_elements_id].shape)

    return change, centroids



# class KMeans(KMeansBase):
#     def __init__(self, n_clusters, max_iters=100, random_state=32, k_max=0.4, k_min=0.01):
#         self.n_clusters = n_clusters
#         self.max_iters = max_iters
#         self.k_max = k_max
#         self.k_min = k_min
#         self.random_state = random_state
#         np.random.seed(random_state)

#     def fit(self, X):
#         # Initialize centroids randomly
#         self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        
#         for _ in range(self.max_iters):
#             # Assign each data point to the nearest centroid
#             labels = self._assign_labels(X)
            
#             # Update centroids
#             new_centroids = self._update_centroids(X, labels)
            
#             # Check for convergence
#             if np.all(self.centroids == new_centroids):
#               break

#             self.centroids = new_centroids

#         for _ in range(self.max_iters):
#             # Assign each data point to the nearest centroid
#             labels = self._assign_labels(X)
            
#             # Update centroids
#             new_centroids = self._update_centroids(X, labels)

#             if np.all(self.centroids == new_centroids):
#               change = self.split_reassign(labels, X)
#               # Check for convergence
#               if not change:
#                 break

#             self.centroids = new_centroids

#         labels = self._assign_labels(X)
#         self.labels_ = labels
#         self.cluster_centers_ = self.centroids
#         return self

#     def split_reassign(self, labels, X):
#       # if a cluster has more than k% of all elements split in two clusters by applying k-means to it in half
#       change = False
#       for i in range(self.n_clusters):
#         if (labels == i).sum() > self.k_max*len(labels):
#           change = True
#           X_ = X[labels == i]
#           kmeans = KMeansBase(2)
#           kmeans.fit(X_)
#           self.centroids[i] = kmeans.centroids[0]
#           self.centroids = np.vstack([self.centroids, kmeans.centroids[1]])
#           self.n_clusters += 1

#       if not change:
#         remove_ids = []
#         for i in range(self.n_clusters):
#           if (labels == i).sum() < self.k_min*len(labels):
#             # assign to the spatially nearest cluster
#             remove_ids.append(i)
        
#         if len(remove_ids) > 0:
#           self.centroids = np.delete(self.centroids, remove_ids, axis=0)
#           self.n_clusters -= len(remove_ids)
#           change = True

#       return change

class KMeans(KMeansBase):
    def __init__(self, n_clusters, max_iters=100, random_state=32, k_max=0.4, k_min=0.01):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.k_min = k_min
        self.random_state = random_state
        np.random.seed(random_state)

    def fit(self, X):
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        
        for _ in range(self.max_iters):
            # Assign each data point to the nearest centroid
            labels = self._assign_labels(X)
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check for convergence
            if np.all(self.centroids == new_centroids):
              break

            self.centroids = new_centroids

        for _ in range(self.max_iters):
            # Assign each data point to the nearest centroid
            labels = self._assign_labels(X)
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)

            if np.all(self.centroids == new_centroids):
              change = self.split_reassign(labels, X)
              # Check for convergence
              if not change:
                break

            self.centroids = new_centroids

        labels = self._assign_labels(X)
        self.labels_ = labels
        self.cluster_centers_ = self.centroids
        return self

    def split_reassign(self, labels, X):
      # if a cluster has more than k% of all elements split in two clusters by applying k-means to it in half
      change = False
      remove_ids = []
      for i in range(self.n_clusters):
        if (labels == i).sum() < self.k_min*len(labels):
          # assign to the spatially nearest cluster
          remove_ids.append(i)
      
      if len(remove_ids) > 0:
        # assign all centroids to the cluster with the most elements
        most_elements = -1
        for i in range(self.n_clusters):
          if (labels == i).sum() > most_elements:
            most_elements = (labels == i).sum()
            most_elements_id = i

        # compute the sigma of all elements in the max centroid
        sigma = np.std(X[labels == most_elements_id], axis=0)
        for i in remove_ids:
          self.centroids[i] = self.centroids[most_elements_id] + np.random.normal(0, 0.01*sigma, self.centroids[most_elements_id].shape)

      return change
