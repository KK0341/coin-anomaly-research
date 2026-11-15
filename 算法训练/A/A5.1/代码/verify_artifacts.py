import sys,re,json,hashlib
sys.path.insert(0,r'E:\异常检测\算法训练\A5.1\代码')
from common import *
report=O/'实验报告.md';txt=report.read_text(encoding='utf8')
for name in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',txt):assert (O/name).exists(),name
scores=read(OUT/'local_scores.csv');assert len(scores)==15100
assert scores.groupby(['method','seed']).size().eq(755).all();assert scores.groupby(['method','seed'])['钱币编号'].nunique().eq(755).all()
assert json.loads((OUT/'verification.json').read_text())['passed'];assert '四个核心问题' in txt and '配准分层与切块边界诊断' in txt
hashes=json.loads((OUT/'input_hashes.json').read_text());assert all(sha(Path(p))==h for p,h in hashes.items())
status={'status':'complete','scope':'A5.1 five-seed minimal validation','seeds':SEEDS,'validation_coins':755,'local_prediction_rows':len(scores),'new_backbone_training':False,'test_access':False,'historical_inputs_unchanged':True,'report_sha256':sha(report),'visual_examples':6,'note':'exploratory true-category study; no parameter or seed selection after full scores; independent confirmation requires new data'}
save(OUT/'COMPLETE.json',status);print(status)
