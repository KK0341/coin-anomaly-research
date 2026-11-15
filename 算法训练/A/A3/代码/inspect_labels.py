from pathlib import Path
import csv,json,collections
import openpyxl
R=Path(__file__).resolve().parents[1]
for d in ['结果','模型','融合']: (R/d).mkdir(parents=True,exist_ok=True)
raw={s:list(csv.DictReader((R.parents[1]/'数据划分'/f).open(encoding='utf-8-sig',newline=''))) for s,f in {'train':'训练集.csv','val':'验证集.csv','test':'测试集.csv'}.items()}
for s,rs in raw.items():
 print(s,dict(collections.Counter(r['正常数字等级'] for r in rs if r['二分类标签']=='0')))
w=openpyxl.load_workbook(R.parents[1]/'袁世凯采集记录.xlsx',read_only=True,data_only=True)
print(w.sheetnames)
ws=w['钱币详细信息'];it=ws.iter_rows(values_only=True);head=next(it);print(head)
data=[dict(zip(head,r)) for r in it]
(R/'结果/excel_grade_rows.json').write_text(json.dumps([{k:r.get(k) for k in head if k and any(x in str(k) for x in ['证书','评级','等级'])} for r in data],ensure_ascii=False,default=str),encoding='utf-8')
w.close()
