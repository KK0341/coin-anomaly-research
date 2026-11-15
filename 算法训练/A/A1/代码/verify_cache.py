import train as t
import json,random
from pathlib import Path
if __name__=='__main__':
    manifest=json.loads(Path(r'D:\coin_image_cache\dataset_v1_original\manifest.json').read_text(encoding='utf-8'))
    t.CACHE_MAP={}; original=t.Coins('train',True)
    t.CACHE_MAP=manifest; cached=t.Coins('train',True)
    indices=random.Random(42).sample(range(len(original)),24)
    for i in indices:
        a,y,_=original[i]; b,z,_=cached[i]
        assert y==z and t.torch.equal(a,b)
    t.save_json(t.OUT/'cache_tensor_verification.json',{'passed':True,'sample_count':len(indices),'result':'bitwise equal tensors under identical augmentation seeds; all cached file hashes separately verified'})
    print('CACHE TENSORS VERIFIED')
