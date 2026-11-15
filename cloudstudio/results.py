"""Export small reviewed result bundles; receive append-only local copies. No model upload."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE = 2 * 1024**2
MAX_TOTAL = 20 * 1024**2
SUFFIXES = {'.json', '.md', '.png', '.jpg', '.jpeg'}
SECRET = re.compile(rb'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')

def sha(data):
    return hashlib.sha256(data).hexdigest()

def safe_name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', value) or value.upper() in {'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))}:
        raise ValueError('Experiment/run names must be safe ASCII letters, digits, underscore or hyphen')
    return value

def relative_file(value):
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(p in {'.', '..'} or '\\' in p or ':' in p or p.endswith((' ', '.')) for p in path.parts):
        raise ValueError('Unsafe relative result path')
    if path.suffix.lower() not in SUFFIXES or path.name.lower() == 'bundle.json':
        raise ValueError('Only metrics JSON, reports MD and selected PNG/JPEG figures are allowed')
    if any(re.search(r'(secret|credential|token|password|\.env)', p, re.I) for p in path.parts):
        raise ValueError('Possible credential file')
    return path

def read_file(root, relative):
    path = root.joinpath(*relative.parts)
    if any(p.is_symlink() for p in [path, *path.parents]) or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Symlink/path escape refused')
    if path.stat().st_size > MAX_FILE:
        raise ValueError(f'Result exceeds 2 MiB: {relative}')
    data = path.read_bytes()
    if SECRET.search(data):
        raise ValueError('Possible credential detected; content suppressed')
    if path.suffix.lower() == '.json':
        json.loads(data)
    elif path.suffix.lower() == '.md':
        data.decode('utf-8-sig')
    elif path.suffix.lower() == '.png' and not data.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('Invalid PNG signature')
    elif path.suffix.lower() in {'.jpg','.jpeg'} and not data.startswith(b'\xff\xd8\xff'):
        raise ValueError('Invalid JPEG signature')
    return data

def verify(folder):
    manifest = folder/'bundle.json'
    if manifest.is_symlink() or manifest.stat().st_size > MAX_FILE:
        raise ValueError('Invalid bundle manifest')
    info = json.loads(manifest.read_text('utf-8'))
    safe_name(info['experiment']); safe_name(info['run_id'])
    dt.date.fromisoformat(info['date'])
    if folder.parts[-3:] != (info['experiment'], info['date'], info['run_id']):
        raise ValueError('Bundle location differs from its manifest')
    entries = info['files']
    if not 1 <= len(entries) <= 20 or len({e['path'].casefold() for e in entries}) != len(entries):
        raise ValueError('Invalid file count or case-insensitive duplicate')
    total = images = 0
    for entry in entries:
        rel = relative_file(entry['path']); data = read_file(folder, rel)
        if len(data) != entry['bytes'] or sha(data) != entry['sha256']:
            raise ValueError('Bundle integrity mismatch')
        total += len(data); images += rel.suffix.lower() in {'.png','.jpg','.jpeg'}
    if total > MAX_TOTAL or images > 8:
        raise ValueError('Bundle exceeds 20 MiB or 8 figures')
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    if actual != {e['path'] for e in entries} | {'bundle.json'}:
        raise ValueError('Unlisted files in result bundle')
    return info

def export(args):
    source = args.run.resolve(); experiment = safe_name(args.experiment)
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
    run_id = now.strftime('%H%M%S')+'-'+uuid.uuid4().hex[:10]
    folder = ROOT/'cloud_results'/experiment/now.date().isoformat()/run_id
    items = []; total = images = 0
    if not 1 <= len(args.file) <= 20:
        raise ValueError('Select 1 to 20 files explicitly')
    for name in args.file:
        rel = relative_file(name); data = read_file(source, rel)
        total += len(data); images += rel.suffix.lower() in {'.png','.jpg','.jpeg'}
        items.append((rel, data))
    if total > MAX_TOTAL or images > 8 or len({str(p).casefold() for p,d in items}) != len(items):
        raise ValueError('Too many bytes, images, or duplicate names')
    commit = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    dirty = bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=no']))
    info = dict(schema=1, experiment=experiment, date=now.date().isoformat(), run_id=run_id,
                exported_at=now.isoformat(), export_checkout_commit=commit, export_checkout_dirty=dirty,
                provenance_note='Export checkout is not proof of the training revision; include cloud_run.json.',
                files=[dict(path=str(p), bytes=len(d), sha256=sha(d)) for p,d in items])
    folder.mkdir(parents=True, exist_ok=False)
    for rel,data in items:
        out=folder.joinpath(*rel.parts);out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('xb') as handle: handle.write(data)
    (folder/'bundle.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    verify(folder)
    print(folder)

def receive(args):
    if args.pull:
        subprocess.run(['git','-C',str(ROOT),'pull','--ff-only'],check=True)
    destination = args.destination.resolve()
    for manifest in sorted((ROOT/'cloud_results').glob('*/*/*/bundle.json')):
        folder = manifest.parent; info = verify(folder)
        target = destination/info['experiment']/info['date']/info['run_id']
        if target.exists():
            previous = verify(target)
            if previous != info:
                raise ValueError(f'Existing result differs; refusing overwrite: {target}')
            print('Already received:',target); continue
        if any(p.is_symlink() for p in [target,*target.parents]):
            raise ValueError('Symlink destination refused')
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(folder,target,dirs_exist_ok=False)
        verify(target)
        print('Received:',target)

def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    exp=sub.add_parser('export');exp.add_argument('--run',type=Path,required=True);exp.add_argument('--experiment',required=True)
    exp.add_argument('--file',action='append',required=True);exp.set_defaults(func=export)
    rec=sub.add_parser('receive');rec.add_argument('--destination',type=Path,default=ROOT/'云端实验结果')
    rec.add_argument('--pull',action='store_true');rec.set_defaults(func=receive)
    args=parser.parse_args();args.func(args)

if __name__=='__main__':main()
