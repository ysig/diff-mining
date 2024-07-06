import numpy as np
import multiprocessing
from multiprocessing import Manager
from joblib.externals.loky.backend.context import get_context

import os
import cv2
import torch
import random
import hashlib
import joblib
import time
from os.path import join
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from safetensors.torch import save_file
from safetensors import safe_open

from skimage.color import rgb2lab
from skimage.feature import hog
from skimage.io import imread

def get_hoglab_single(path):
    # read image
    image = imread(path)

    # hog features
    hog_feats = hog(image, orientations=31, pixels_per_cell=(8, 8), cells_per_block=(8, 8), channel_axis=-1, feature_vector=False)
    hog_feats = hog_feats.reshape(hog_feats.shape[0], hog_feats.shape[1], 8*8*31)

    # lab features
    lab_feats = rgb2lab(image)
    lab_feats = torch.from_numpy(lab_feats).permute(2, 0, 1).unsqueeze(1)[1:3]
    lab_feats = F.unfold(lab_feats, kernel_size=(64, 64), stride=(8, 8))

    # resize each patch to 8x8
    K = lab_feats.shape[-1]
    lab_feats = lab_feats.permute(2, 0, 1).reshape(K, 2, 64, 64)
    lab_feats = F.interpolate(lab_feats, size=(8, 8), mode='bilinear', align_corners=False)
    lab_feats = (lab_feats.reshape(K, 2*8*8) + 128.0)/255.0

    lab_feats = lab_feats.reshape(hog_feats.shape[0], hog_feats.shape[1], 2*8*8).numpy().squeeze()
    x = np.concatenate([hog_feats, lab_feats], axis=-1)
    return x.transpose(1, 0, 2)

class DatasetHOG(Dataset):
    def __init__(self, paths, cache_path=None):
        self.paths = paths
        self.cache_path = cache_path
        if self.cache_path is not None:
            os.makedirs(self.cache_path, exist_ok=True)

        valid_paths = []
        for idx, path in enumerate(self.paths):
            image = cv2.imread(path)
            if image is not None:
                valid_paths.append(path)
            else:
                print(f'Path {path} is invalid')
        self.paths = valid_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        hash_path = os.path.abspath(image_path)
        hash_path = hash_path.replace('/', '_')
        fpath = join(self.cache_path, hash_path + '.npy')
        if not os.path.isfile(fpath):
            feat = get_hoglab_single(image_path)
            np.save(fpath, feat)
        else:
            feat = np.load(fpath)

        feats = np.squeeze(np.load(fpath))
        feats = normalize(feats)
        return torch.from_numpy(feats).unsqueeze(0).half(), image_path

def normalize(feats):
    # mean_feats = np.mean(feats, axis=-1, keepdims=True)
    # feats -= mean_feats
    feat_norm = np.linalg.norm(feats, axis=-1, keepdims=True)
    # feat_norm[feat_norm == 0] = 1
    feats = feats / feat_norm
    return feats

def collate_fn(batch):
    return torch.cat([b[0] for b in batch], dim=0), [b[1] for b in batch]

def pre_safetensors(paths, cache_path, safetensors_path, tag, num_splits=4, batch_size=16, recompute=False):
    safetensors_path = join(safetensors_path, f'{tag}')
    paths_out_fpath = join(safetensors_path, f'{tag}_paths.pkl')
    if recompute or not os.path.isfile(paths_out_fpath):
        dataloader = DataLoader(DatasetHOG(paths, cache_path), batch_size=batch_size, num_workers=min(16, batch_size), collate_fn=collate_fn, shuffle=False)
        idx, paths_out, safetensors = 0, [], {}
        for data, pts in tqdm(dataloader, desc='safetensors'):
            safetensors[';;'.join(pts)] = data
            if len(safetensors) == len(dataloader)//num_splits or idx == len(dataloader)-1:
                paths_out.append(join(safetensors_path, f'{idx}.safetensors'))
                os.makedirs(safetensors_path, exist_ok=True)
                save_file(safetensors, paths_out[-1])
                safetensors = {}
                idx += 1
        
        joblib.dump(paths_out, paths_out_fpath)

    return joblib.load(paths_out_fpath)

def accumulate(queue, top_k, lenw, sorted_buffer, ret_ws):
    while True:
        obj = queue.get()
        if obj is None:
            break

        for i in tqdm(range(lenw), desc=f'inner-{top_k}', disable=True):
            sorted_buffer[i] = sorted(sorted_buffer[i] + obj[i], key=lambda x: x[0], reverse=True)[:top_k]

def make_bbox(i, dim):
    a, b = np.unravel_index(i, dim)
    return (a*8, b*8)

@torch.no_grad()
def dense_search_cuda(w, sft_paths, top_k=50, disable_tqdm=True, ret_ws=False, activate_pdb=False, fold=None, only_pos=False, device_id='cuda'):
    device = torch.device(device_id)
    w = torch.from_numpy(w).to(device).half()
    K = w.shape[0]
    with Manager() as manager:
        context = get_context('loky')
        queue = context.Queue()
        sorted_buffer = manager.list([[] for _ in range(K)])
        p = context.Process(target=accumulate, args=(queue, top_k, len(w), sorted_buffer, ret_ws))
        p.start()
        for path_id, sft_path in enumerate(sft_paths):
            with safe_open(sft_path, framework="pt", device=(device_id if device_id in {'cuda', 'cpu'} else int(device_id.replace('cuda:', '')))) as f:
                pbar = tqdm(f.keys(), desc='dense-search', disable=disable_tqdm)
                for key in pbar:
                    paths = key.split(';;')
                    data = f.get_tensor(key).to(device).half()
                    B, W, H, C = data.shape

                    data = data.reshape(B, W*H, C)
                    scores = (data.reshape(B*W*H, C).unsqueeze(1) * w.unsqueeze(0)).sum(dim=-1)
                    scores = (data.reshape(B*W*H, C).unsqueeze(1) * w.unsqueeze(0)).sum(dim=-1)
                    scores = scores.reshape(B, W*H, K).permute(2, 0, 1)

                    if fold is not None:
                        mask = torch.zeros(B, W*H, device=device)
                        torch.manual_seed(path_id)
                        indices = torch.stack([torch.randperm(W*H, device=device)[:(fold[0]*(W*H))//fold[1]] for _ in range(B)])
                        mask = mask.scatter(1, indices, 1)
                        scores = scores * mask.unsqueeze(0)
                        
                    scores = scores.reshape(K*B, W*H)
                    indexes = torch.topk(scores, 1, dim=-1)

                    while queue.qsize() > 3: 
                        pbar.set_postfix({'queue': queue.qsize()})
                        pbar.set_description('throttling')
                        time.sleep(0.1)

                    pbar.set_postfix({'queue': queue.qsize()})
                    pbar.set_description('')

                    L = indexes.indices.shape[-1]
                    indexes_numpy, values_numpy = indexes.indices.reshape(K, B, L).cpu().numpy(), indexes.values.reshape(K, B, L).cpu().numpy()
                    queue.put(
                        [
                            [
                                (values_numpy[k, b, l],
                                 make_bbox(indexes_numpy[k, b, l], (W, H)),
                                 paths[b]
                                ) + ((data[b, indexes_numpy[k, b, l]].cpu().numpy(),) if ret_ws else tuple())
                                for b in range(B) 
                                for l in range(L)
                                if not only_pos or values_numpy[k, b, l] > 0
                            ]
                            for k in range(K)
                        ]
                    )
        queue.put(None)
        p.join()
        assert all(map(len, sorted_buffer))
        return list(sorted_buffer)

@torch.no_grad()
def random_sample(sft_paths, fold=None, num_samples=10000):
    paths = list(sft_paths)
    random.shuffle(paths)
    samples_per_block = num_samples//len(paths)
    for sft_path, n_samples in zip(paths, [samples_per_block]*len(paths)): 
        with safe_open(sft_path, framework="pt", device="cpu") as f:
            # divide n_samples per keys and make sure that it is not zero
            n_samples_per_key = n_samples//len(f.keys())
            n_samples_per_key = max(1, n_samples_per_key)
            keys = list(f.keys())
            random.shuffle(keys)
            for key in keys:
                paths = key.split(';;')
                data = f.get_tensor(key).half()
                B, H, W, C = data.shape
                data = data.reshape(B*H*W, C)

                if fold is not None:
                    torch.manual_seed(0)
                    indices = torch.randperm(B*H*W, device='cpu')[:(fold[0]*(B*H*W))//fold[1]].tolist()
                    for i in random.sample(indices, n_samples_per_key):
                        yield data[i].numpy()
                else:
                    for i in random.sample(range(B*H*W), n_samples_per_key):
                        yield data[i].numpy()


if __name__ == "__main__":
    path = '/home/isig/diff-geo-mining/doersch/geo-base-data/Brazil/gt--Brazil___0SwfTQj2O0fU_u2OHOXfA_315.jpg'
    print(get_hoglab_single(path).shape)
