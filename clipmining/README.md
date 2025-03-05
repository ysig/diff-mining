# CLIP Baseline
This baseline is not documented on the main paper, due to space. It is constructed by taking CLIP (specifically StreetCLIP) and rescaling an input image to 334 x 334 (by center cropping) and then computing I) a ranking of each patch and II) clustering them top-1000 by features per position.
It is exactly our algorithm from the main paper but adapted to CLIP.

## I. Ranking each patch
To rank each patch we try both cosine similarity to the target label and normalize it through the null conditioning, constructing the following [ranking scores](https://github.com/ysig/diff-mining/blob/b5f82056aaa9fce5154fc2f118d9e58cf5e9dbcc/clipmining/ranking.py#L78-L81):
a) **Similarity**: `sim(patch, f"{country}")`.
b) **Difference**: `sim(patch, f"{country}") - sim(patch, f"") `.  

## II. Clustering 
To extract features for clustering we simply [rescale and upscale token features and take their l2-normalized average](https://github.com/ysig/diff-mining/blob/22808cc6f9f1a773fe8c3ef9c27a9d3de2687430/clipmining/ranking.py#L99-L107), that corresponds to the input patch, similar to what we do for our dift features.

## Results
Results for our 10 mined countries can be located [on our suppmat](https://diff-mining.github.io/supmat/clip.html).
Although the speed of this algorithm is extremely fast: 30 minutes per country on vector parallelization of 32 cpus**(!)**, the results are unfortunately not that satisfying.
Note that, to the best of our knowledge, because of the [learned positional embedding of clip](https://github.com/huggingface/transformers/blob/6af0854efa3693e0b38c936707966685ec3d0ae8/src/transformers/models/clip/modeling_clip.py#L185) this method [can't be extended to arbitrary size images](https://discuss.huggingface.co/t/proper-way-to-handle-non-square-images-with-clip/32813).
