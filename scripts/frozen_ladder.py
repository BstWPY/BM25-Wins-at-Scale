#!/usr/bin/env python3
"""Verify and export the paper's frozen document sets without an LLM or network."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / 'data_manifests' / 'frozen'


def id_hash(ids):
    """Historical convention: UTF-8 IDs joined by LF, without a trailing LF."""
    return hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()


def read_ids(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8', newline='') as stream:
        ids = stream.read().splitlines()
    if any(not d or d != d.strip() for d in ids):
        raise ValueError(f'Blank or padded identifier in {path.name}')
    return ids


def load_ladder(root=DEFAULT_ROOT):
    root = Path(root)
    bedrock = read_ids(root / 'bedrock_dsids.txt')
    ocean = read_ids(root / 'ocean_order.txt.gz')
    refs = json.loads((root / 'reference_checks.json').read_text(encoding='utf-8'))
    if len(bedrock) != 1144 or len(ocean) != 510815:
        raise ValueError('Incorrect bedrock or background size')
    combined = bedrock + ocean
    if len(set(combined)) != len(combined):
        raise ValueError('Repeated identifier or bedrock/background overlap')
    if bedrock != sorted(bedrock):
        raise ValueError('Bedrock must use the historical sorted order')
    sizes = json.loads((root / 'tier_sizes.json').read_text(encoding='utf-8'))
    if len(refs) != 28 or [r['N_docs'] for r in refs] != sizes:
        raise ValueError('Expected exactly the frozen 28-tier reference sequence')
    if sizes != sorted(set(sizes)) or sizes[0] != 1144 or sizes[-1] != 511959:
        raise ValueError('Invalid tier sizes')
    # Historical set references are shipped separately from the recovered order.
    historical = json.loads((root.parent / 'tier_manifest_summary.json').read_text(encoding='utf-8'))
    historical = {r['N_docs']: r['manifest_sha256'] for r in historical}
    checked = []
    for ref in refs:
        n = ref['N_docs']
        order = combined[:n]
        set_hash, order_hash = id_hash(sorted(order)), id_hash(order)
        if set_hash != historical.get(n) or set_hash != ref['expected_sha256']:
            raise ValueError(f'Historical document set mismatch at N={n}')
        if order_hash != ref['order_sha256']:
            raise ValueError(f'Frozen recovered order mismatch at N={n}')
        checked.append({'N_docs': n, 'set_sha256': set_hash, 'order_sha256': order_hash,
                        'reference_N_chunks': ref['reference_N_chunks']})
    return bedrock, ocean, checked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    ap.add_argument('--tier', type=int, help='Optionally export one historical tier')
    ap.add_argument('--out', type=Path, help='New output directory; existing paths are refused')
    args = ap.parse_args()
    bedrock, ocean, checks = load_ladder(args.root)
    if (args.tier is None) != (args.out is None):
        ap.error('--tier and --out must be used together')
    if args.tier is not None:
        ref = next((r for r in checks if r['N_docs'] == args.tier), None)
        if ref is None:
            ap.error('Requested tier is not one of the 28 historical tiers')
        args.out.mkdir(parents=True, exist_ok=False)
        order = bedrock + ocean[:args.tier-len(bedrock)]
        (args.out / 'ordered_dsids.txt').write_bytes(('\n'.join(order)+'\n').encode('utf-8'))
        manifest = {**ref, 'sha256': ref['set_sha256'], 'dsids': sorted(order),
                    'order_status': 'recovered from archived input and original algorithm'}
        (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'verified_tiers': len(checks), 'bedrock_docs': len(bedrock),
                      'background_docs': len(ocean), 'strictly_nested': True,
                      'historical_set_hashes_match': True, 'exported_tier': args.tier}))


if __name__ == '__main__':
    main()
