import requests,pathlib,json,zipfile,hashlib,time
O=pathlib.Path('E:/异常检测/算法训练/A10_预训练骨干对照实验'); s=requests.Session()
r=s.get('https://api.github.com/repos/facebookresearch/dinov2/commits/main',timeout=30); r.raise_for_status(); commit=r.json()['sha']; print('commit',commit,flush=True)
u='https://codeload.github.com/facebookresearch/dinov2/zip/'+commit
r=s.get(u,timeout=90); r.raise_for_status(); z=O/'来源/dinov2.zip'; z.write_bytes(r.content)
with zipfile.ZipFile(z) as f: f.extractall(O/'来源')
u='https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'; p=O/'模型/dinov2_vits14_pretrain.pth'
with s.get(u,timeout=(20,60),stream=True) as r:
 r.raise_for_status()
 with p.open('wb') as f:
  for chunk in r.iter_content(1024*1024): f.write(chunk)
info={'variant':'A10b','reason':'DINOv3 official HF HTTP401 restricted; Meta requires access application; no local torch/HF cache; timm not installed','repo':'https://github.com/facebookresearch/dinov2','commit':commit,'source_dir':str(O/'来源'/('dinov2-'+commit)),'weight_url':u,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'weight_bytes':p.stat().st_size}
(O/'来源/provenance.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf8'); print(info,flush=True)
