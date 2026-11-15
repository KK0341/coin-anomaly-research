"""Package only the existing six-class splits; never repartition data."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from collections import Counter

CODES = ['92', '97', '98', '95', '94', '91']

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--b0', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    source = a.b0 / '数据缓存'
    original = source / 'manifest.json'
    manifest = json.loads(original.read_text('utf8'))
    a.output.mkdir(parents=True, exist_ok=True)
    dest = a.output / 'b5-assets.tar'
    if dest.exists():
        raise FileExistsError(dest)
    selected, files, counts, all_ids = {}, {}, {}, set()
    for split, expected in [('train', 1583), ('val', 198), ('test', 198)]:
        rows = [r for r in manifest[split] if r['二分类标签'] == '1' and r['异常代码'] in CODES]
        assert len(rows) == expected
        selected[split] = []
        for r in rows:
            coin = r['钱币编号']
            assert coin not in all_ids, ('duplicate coin', coin)
            all_ids.add(coin)
            assert r['images'] == [f'images/{coin}_0.png', f'images/{coin}_1.png']
            selected[split].append({k: r[k] for k in ['钱币编号', '异常代码', '二分类标签', 'images']})
            for name in r['images']:
                files[name] = source / name
        counts[split] = dict(Counter(r['异常代码'] for r in rows))
    files.update({'a8.pt': source / 'a8.pt', 'b0.pt': a.b0 / 'best.pt', 'b0_val_metrics.json': a.b0 / 'val_metrics.json', 'b0_config.json': a.b0 / 'config.json'})
    mp = a.output / 'manifest.json'
    mp.write_text(json.dumps(selected, ensure_ascii=False, indent=2), 'utf8')
    files['manifest.json'] = mp
    hashes = {name: digest(path) for name, path in files.items()}
    audit = json.loads((source / 'data_audit.json').read_text('utf8'))
    assert hashes['a8.pt'] == audit['a8.pt'], 'A8 checksum mismatch'
    record = {'original_manifest_sha256': digest(original), 'counts': counts, 'coins': len(all_ids), 'images': len(all_ids)*2, 'sha256': hashes}
    hp = a.output / 'assets.json'
    hp.write_text(json.dumps(record, ensure_ascii=False, indent=2), 'utf8')
    with tarfile.open(dest.with_suffix('.tmp'), 'w') as tar:
        for name, path in files.items():
            tar.add(path, arcname=name, recursive=False)
        tar.add(hp, arcname='assets.json', recursive=False)
    dest.with_suffix('.tmp').replace(dest)
    (a.output / 'archive_sha256.txt').write_text(digest(dest) + '  b5-assets.tar\n', 'utf8')
    print(json.dumps({'counts': counts, 'coins': len(all_ids), 'images': len(all_ids)*2, 'archive_bytes': dest.stat().st_size}))

if __name__ == '__main__':
    main()
