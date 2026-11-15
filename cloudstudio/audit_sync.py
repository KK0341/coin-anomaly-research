"""Audit the prospective Git payload without staging or printing secret values."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import argparse
from results import verify

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--staged', action='store_true', help='Audit exact index blobs before commit')
    args = parser.parse_args()
    paths = set()
    for options in ([['--cached']] if args.staged else [['--cached'], ['--others', '--exclude-standard']]):
        raw = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '-z', *options])
        paths.update(x.decode('utf-8') for x in raw.split(b'\0') if x)
    issues = []
    total = 0
    credential = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
    for relative in sorted(paths):
        path = ROOT / relative
        if not path.is_file():
            continue
        data = subprocess.check_output(['git','-C',str(ROOT),'show',':'+relative]) if args.staged else path.read_bytes()
        if args.staged and relative.startswith('cloud_results/') and data != path.read_bytes():
            issues.append([relative,'staged result differs from verified worktree bytes'])
        size = len(data)
        total += size
        if size > 5 * 1024**2:
            issues.append([relative, 'over 5 MiB review limit'])
        allowed = {'.py', '.md', '.json', '.txt', '.sh', '.ps1'}
        if relative.startswith('cloud_results/'):
            allowed = {'.json','.md','.png','.jpg','.jpeg'}
            if size > 2 * 1024**2: issues.append([relative,'result over 2 MiB'])
        if path.suffix.lower() not in allowed and path.name not in {'.gitignore', '.gitattributes'}:
            issues.append([relative, 'unexpected file type'])
        if re.search(r'(?:^|/)(?:\.env|.*(?:secret|credential|password|token)|id_rsa|id_ed25519)',relative,re.I):
            issues.append([relative,'possible credential filename'])
        if size <= 5 * 1024**2 and credential.search(data):
            issues.append([relative, 'possible credential; value suppressed'])
    manifest = json.loads((ROOT / 'cloudstudio/source_sha256.json').read_text(encoding='utf-8'))
    lf_manifest = json.loads((ROOT / 'cloudstudio/source_lf_sha256.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        data = subprocess.check_output(['git','-C',str(ROOT),'show',':'+name]) if args.staged else (ROOT/name).read_bytes()
        if hashlib.sha256(data).hexdigest() not in {expected, lf_manifest[name]}:
            issues.append([name, 'original source changed'])
    for bundle in (ROOT/'cloud_results').glob('*/*/*/bundle.json'):
        try: verify(bundle.parent)
        except Exception as exc: issues.append([str(bundle.relative_to(ROOT)),str(exc)])
    for name in paths:
        if name.startswith('cloud_results/') and name != 'cloud_results/README.md':
            parts=Path(name).parts
            if len(parts)<5 or not (ROOT.joinpath(*parts[:4])/'bundle.json').is_file():
                issues.append([name,'result file has no bundle manifest'])
    print(json.dumps({'candidate_files': len(paths), 'total_bytes': total, 'issues': issues,
                      'note': 'Pattern scan only; review staged diff before first push.'}, ensure_ascii=False, indent=2))
    return bool(issues)

if __name__ == '__main__':
    raise SystemExit(main())
