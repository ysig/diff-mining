from huggingface_hub import snapshot_download
for model in ['xray', 'places', 'g3', 'ftt', 'cardb']:
    snapshot_download(repo_id=f"diff-mining/{model}", local_dir=f"models/{model}", repo_type='model')
  
