"""Read-only environment check; no model, data, download or training."""
import importlib.metadata as md
import json
import sys

def main():
    report = {'python': sys.version, 'executable': sys.executable, 'packages': {}}
    for name in ['torch', 'torchvision', 'numpy', 'Pillow', 'scikit-learn', 'matplotlib']:
        try:
            report['packages'][name] = md.version(name)
        except md.PackageNotFoundError:
            report['packages'][name] = None
    try:
        import torch
        import torchvision
        report.update(cuda_available=torch.cuda.is_available(), cuda=torch.version.cuda,
                      devices=[{'name': torch.cuda.get_device_name(i),
                                'memory_GiB': round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2),
                                'capability': torch.cuda.get_device_capability(i)}
                               for i in range(torch.cuda.device_count())])
    except Exception as exc:
        report['import_error'] = str(exc)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get('cuda_available') and all(report['packages'].values()) else 1

if __name__ == '__main__':
    raise SystemExit(main())
