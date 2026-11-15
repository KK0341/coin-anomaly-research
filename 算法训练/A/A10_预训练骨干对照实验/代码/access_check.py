import requests,json,pathlib
O=pathlib.Path('E:/异常检测/算法训练/A10_预训练骨干对照实验/来源')
s=requests.Session(); results={}
for name,url in [('dinov3_hub','https://raw.githubusercontent.com/facebookresearch/dinov3/main/dinov3/hub/backbones.py'),('dinov3_hf','https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m/resolve/main/model.safetensors'),('dinov2_hub','https://raw.githubusercontent.com/facebookresearch/dinov2/main/dinov2/hub/backbones.py')]:
 try:
  r=s.get(url,timeout=35,stream=True); results[name]={'url':url,'status':r.status_code}
  if name.endswith('hub') and r.ok: (O/(name+'.py')).write_text(r.text,encoding='utf8')
  elif not r.ok: results[name]['reason']=r.text[:500]
  r.close()
 except Exception as e: results[name]={'error':str(e)}
 print(name,results[name],flush=True)
(O/'access_checks.json').write_text(json.dumps(results,indent=2),encoding='utf8')
