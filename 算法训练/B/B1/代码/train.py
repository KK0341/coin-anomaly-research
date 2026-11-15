import runpy,sys
from pathlib import Path
if __name__=="__main__":
    target=Path(__file__).resolve().parents[2]/"B0/代码/train.py"
    sys.argv=[str(target),"B1"]
    runpy.run_path(str(target),run_name="__main__")
