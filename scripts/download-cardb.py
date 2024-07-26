import os
import zipfile
from huggingface_hub import snapshot_download

snapshot_download(repo_id="diff-mining/cardb", local_dir="datasets/cardb", repo_type='dataset')
for root, dirs, files in os.walk("datasets/cardb"):
    for file in files:
        if file.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(root, file), 'r') as zip_ref:
                zip_ref.extractall(root)
                os.remove(os.path.join(root, file))
