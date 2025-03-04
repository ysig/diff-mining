import os
import PIL
import operator
import torch
import joblib
import pandas as pd
import numpy as np
from torchvision import transforms
from os.path import join
from tqdm import tqdm, trange
from PIL import Image

from collections import defaultdict
from torch.nn.functional import interpolate
from transformers import CLIPProcessor, CLIPVisionModel, CLIPTextModel, CLIPTextModelWithProjection, CLIPModel
from sklearn.cluster import KMeans
from utils import pool, sort, get_non_overlapping, get_top_k, make_grid

class CLIPRankCluster(object):
    def __init__(self, dataset_path='/home/isig/dataset/geo-base-data', cache_path='clip', mode='diff'):
        self.model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
        self.text_model = CLIPTextModelWithProjection.from_pretrained("geolocal/StreetCLIP")
        self.processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
        self.processor.image_processor.do_center_crop = False
        self.dataset_path = dataset_path
        self.load_paths_geo(dataset_path)
        self.cache_path = join(cache_path, mode)
        self.mode = mode

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
        print(self.parent.keys())

    def categories(self):
        return sorted(list(self.parent.keys()))

    def get_seeds(self, c):
        return [path[0] for path in self.country_path[c] if path[1]]

    def process_text(self, text):
        return self.processor(text=text, return_tensors="pt", padding=True)

    def process_image(self, image):
        image = self.processor(images=image, return_tensors="pt")
        return image

    def project_image(self, image):
        outputs = self.model.vision_model(**image).last_hidden_state
        tokens = self.model.vision_model.post_layernorm(outputs[:, 1:, :])
        outputs = self.model.visual_projection(tokens)
        return outputs

    def project_text(self, text):
        text_embeds = self.text_model(**text).text_embeds
        return text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    def dot_text_image(self, path, text_embeds, vision_encoding, input_size, pw, k_per_image, kx=64, ky=64):
        scores = torch.einsum("ke,bpe->bpk", text_embeds, vision_encoding / vision_encoding.norm(p=2, dim=-1, keepdim=True))
        dims = scores.shape[-2] // pw, pw
        scores = scores_2d = scores.view(-1, dims[0], dims[1], 2)
        scores_2d = scores_2d.softmax(dim=-1)[:, :, :, 0].squeeze(0)
        scores = torch.nn.functional.interpolate(scores.permute(0, 3, 1, 2), size=input_size, mode="bilinear", align_corners=False)
        if self.mode == 'diff':
            scores = pool(scores[:, 0].unsqueeze(0), kx, ky).squeeze(0).squeeze(0) - pool(scores[:, 1].unsqueeze(0), kx, ky).squeeze(0).squeeze(0)
        elif self.mode == 'sim':
            scores = pool(scores, kx, ky)[:,0].squeeze(0)
        else:
            raise ValueError(f'Invalid mode {mode}')

        flattened_scores = scores.flatten()
        top_k = min(k_per_image * 100, flattened_scores.numel())
        _, top_indices = torch.topk(flattened_scores, k=top_k, largest=True)
        rows = top_indices // scores.shape[1]
        cols = top_indices % scores.shape[1]
        df_data = [
            (path, row.item(), col.item(), row.item() + kx, col.item() + ky, scores[row, col].item())
            for row, col in zip(rows, cols)
        ]
        
        df = pd.DataFrame(df_data, columns=['seed', 'x_start', 'y_start', 'x_end', 'y_end', 'D'])
        result = get_non_overlapping(df, k_per_image=k_per_image)
        vision_encoding = vision_encoding.squeeze(0).permute(1, 0).reshape(-1, dims[0], dims[1])
        vision_encoding = interpolate(vision_encoding.unsqueeze(0), size=input_size, mode='bilinear').squeeze(0)
        embeds = []
        for i in range(len(result)):
            # crop and take the average
            x_start, y_start, x_end, y_end = result.iloc[i][['x_start', 'y_start', 'x_end', 'y_end']]
            crop = vision_encoding[:, x_start:x_end, y_start:y_end]
            crop = crop.flatten(start_dim=1).mean(dim=1)
            crop = crop / crop.norm(p=2, dim=-1, keepdim=True)
            embeds.append(crop.cpu().numpy())

        return result, embeds

    def load_image(self, path):
        real_image = PIL.Image.open(path).convert('RGB')
        real_image = transforms.CenterCrop(512)(real_image)
        return real_image

    @torch.no_grad()
    def rank(self, country, k_per_image, kx=64, ky=64):
        paths = self.get_seeds(country)
        text_embeds = self.project_text(self.process_text([country, ""]))
        dfs = []
        embeds = []
        for path in tqdm(paths, desc=f'Ranking [{country}]'):
            real_image = self.load_image(path)
            clip_img = self.process_image(real_image)
            vision_enc = self.project_image(clip_img)
            df, emb = self.dot_text_image(path, text_embeds, vision_enc, (real_image.height, real_image.width), 24, k_per_image, kx, ky)
            dfs.append(df)
            embeds += emb

        return pd.concat([df for df in dfs], axis=0), embeds

    def cluster(self, df, embeds, num_clusters=32):
        clustering = KMeans(n_clusters=num_clusters, random_state=10)
        clustering.fit(embeds)
        clusters = defaultdict(list)
        for (i, l) in enumerate(clustering.labels_):
            x_start, y_start, x_end, y_end = df.iloc[i][['x_start', 'y_start', 'x_end', 'y_end']]
            pil = self.load_image(df.iloc[i]['seed']).crop((y_start, x_start, y_end, x_end))
            ds = df.iloc[i]['D']

            path__ = os.path.split(df.iloc[i]['seed'])[1]
            ext = os.path.splitext(path__)[1]
            ids = path__.replace(ext, '_') + f'{x_start}-{y_start}-{x_end}-{y_end}'
            
            clusters[l].append((pil, ds, ids, embeds[i], df.iloc[i]['seed']))

        for k, vs in clusters.items():
            clusters[k] = [(a, b, c, e) for a, b, c, d, e in sorted(vs, key=lambda x: np.linalg.norm(x[-2] - clustering.cluster_centers_[k]))]

        return sorted([(vs, np.median([v[1] for v in vs]), ) for _, vs in clusters.items()], key=operator.itemgetter(1), reverse=True)

    def clustering(self, k_per_image=5, k=1000, num_clusters=32, hard_limit=6):
        dfs = {}
        cache_path = join(self.cache_path, 'dfs')
        os.makedirs(cache_path, exist_ok=True)
        figure_dir = join(self.cache_path, 'figures')
        os.makedirs(figure_dir, exist_ok=True)
        for country in tqdm(self.categories(), desc='Clustering'):
            fp = os.path.join(cache_path, country + '.pkl')
            try:
                df, embeds = joblib.load(fp)
            except Exception:
                os.makedirs(cache_path, exist_ok=True)
                (df, embeds) = self.rank(country, k_per_image=k_per_image)
                joblib.dump((df, embeds), fp)

            df['index'] = np.arange(len(df))
            df = get_top_k(df, key='D', k=k, randomize=False)
            index = df['index'].iloc[:k].tolist()
            df = df.drop(columns=['index'])
            embs = [embeds[i] for i in index] 

            clusters = self.cluster(df, embs, num_clusters=num_clusters)
            local_dir = join('images', 'clusters', country)
            parent_ = join(self.cache_path, local_dir)
            os.makedirs(parent_, exist_ok=True)

            # column should be the cluster id
            images_grid = []
            for i in range(num_clusters):
                images_row = []
                for j, (pil, ds, idd, _) in enumerate(clusters[i][0]):
                    pil.save(join(parent_, f'{i}-{j}-{num_clusters}_{idd}.png'))
                    if j < hard_limit:
                        images_row.append(pil.convert('RGB'))
                images_grid.append(images_row)

            make_grid(images_grid, horizontal_spacing=2, vertical_spacing=4).save(join(figure_dir, f'{country}.png'))


if __name__ == "__main__":
    # argparser for sim
    import argparse
    args = argparse.ArgumentParser(description="Ranking")
    args.add_argument("--dataset", type=str, default='/home/isig/dataset/geo-base-data')
    args.add_argument("--cache", type=str, default='clip')
    args.add_argument("--mode", type=str, default='diff', choices=['diff', 'sim'])
    args = args.parse_args()
    rank_cluster = CLIPRankCluster(dataset_path=args.dataset, cache_path=args.cache, mode=args.mode)
    rank_cluster.clustering(k_per_image=5, k=1000, num_clusters=32, hard_limit=6)
