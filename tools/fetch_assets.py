"""Explicit model download utility; never downloads CTs or patient metadata."""
import argparse,hashlib,json,os,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--profile',choices=['fp32','fp16','compact-fp16','trt-fp16'],required=True)
    p.add_argument('--destination',type=pathlib.Path,required=True);p.add_argument('--repo',default='Martin123456qwe/radar-local-deploy');a=p.parse_args()
    dest=a.destination.resolve();dest.mkdir(parents=True,exist_ok=True);os.environ['HF_HOME']=str(dest/'.hf_cache')
    sys.path.insert(0,str(ROOT));from run import required
    from huggingface_hub import hf_hub_download
    manifest=json.loads((ROOT/'assets_manifest.json').read_text(encoding='utf-8'));external=json.loads((ROOT/'external_assets.json').read_text(encoding='utf-8'))
    for key in required(a.profile):
        info=external.get(key,manifest.get(key));target=dest/key
        if target.exists() and sha(target)==info['sha256']:continue
        repo=info.get('repo_id',a.repo);filename=info.get('filename',key)
        if key in external:
            downloaded=pathlib.Path(hf_hub_download(repo_id=repo,filename=filename,local_dir=str(dest/'.official_source')))
            import shutil
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(downloaded,target)
        else:hf_hub_download(repo_id=repo,filename=filename,local_dir=str(dest))
        if sha(target)!=info['sha256']:raise RuntimeError('Downloaded hash mismatch: '+key)
    print('Required model assets verified. No images or clinical records downloaded.')
if __name__=='__main__':main()
