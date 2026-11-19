"""Download UCI Covertype and construct the paper's 94 binary features."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import urllib.request
import numpy as np

GZIP_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'
EXPECTED_SHA256 = '614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771'

def encode(raw):
    # Ten continuous columns -> 50 one-hot bins; retain 44 binary indicators.
    edges = np.quantile(raw[:, :10], [0.2, 0.4, 0.6, 0.8], axis=0)
    x = np.zeros((len(raw), 94), dtype=np.uint8)
    for j in range(10):
        bins = np.searchsorted(edges[:, j], raw[:, j], side='right')
        x[np.arange(len(raw)), 5*j + bins] = 1
    x[:, 50:] = raw[:, 10:54].astype(np.uint8)
    y = raw[:, 54].astype(np.int64) - 1
    return x, y, edges

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='data/covertype.npz')
    p.add_argument('--source', help='Existing covtype.data.gz; skips download')
    a = p.parse_args()
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    source = Path(a.source) if a.source else out.parent / 'covtype.data.gz'
    if not source.exists():
        print('Downloading public UCI Covertype data...', flush=True)
        urllib.request.urlretrieve(GZIP_URL, source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError('Source checksum differs from the verified UCI download; inspect the source file.')
    with gzip.open(source, 'rt') as f:
        raw = np.loadtxt(f, delimiter=',', dtype=np.float64)
    if raw.shape != (581012, 55):
        raise ValueError(f'Unexpected dataset shape {raw.shape}')
    x, y, edges = encode(raw)
    np.savez_compressed(out, x=x, y=y, edges=edges)
    metadata = {'url': GZIP_URL, 'sha256': digest,
                'rows': len(y), 'features': 94, 'classes': 7,
                'quantiles': 'Full-dataset quintiles; ties assigned to upper bin; no label use.'}
    out.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    print(f'Saved {out}: {x.shape}, labels {np.unique(y).tolist()}', flush=True)

if __name__ == '__main__':
    main()
