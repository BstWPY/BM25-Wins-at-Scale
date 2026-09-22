#!/usr/bin/env python3
"""Materialize a verified frozen tier from the pinned archived corpus metadata.

No model, API key, or agent is used. Existing output directories are refused.
The source Parquet SHA-256 and the upstream question/scaffold bytes are checked
before any output is written. Results describe reconstruction, not a proof of
byte equality with unavailable historical chunk files.
"""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from frozen_ladder import DEFAULT_ROOT, id_hash, load_ladder

CORPUS_SHA256 = '41d8191e44616227a9be173a6e4a09340b873a07a8f03e3f283ecbde85547cd6'
SCAFFOLDS = {'__scaffold_company_overview': 'generated_data/company_overview.md',
             '__scaffold_initiatives': 'generated_data/initiatives.md'}


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def chunk_body(text, enc):
    toks = enc.encode(text, disallowed_special=())
    if len(toks) <= 1200:
        return [text]
    out = []
    for i in range(0, len(toks), 1100):
        out.append(enc.decode(toks[i:i+1200]))
        if i+1200 >= len(toks):
            break
    return out


def materialize(corpus_meta, benchmark_root, out, tier=42587, frozen_root=DEFAULT_ROOT):
    out, benchmark_root = Path(out), Path(benchmark_root)
    if out.exists():
        raise ValueError('Output already exists; choose a new directory')
    import pyarrow.parquet as pq
    import tiktoken
    bedrock, ocean, checks = load_ladder(frozen_root)
    ref = next((r for r in checks if r['N_docs'] == tier), None)
    if ref is None:
        raise ValueError('Only historical tier sizes are supported')
    provenance = json.loads((Path(frozen_root)/'benchmark_source.json').read_text(encoding='utf-8'))
    for entry in provenance['upstream']['files']:
        if file_hash(benchmark_root/entry['path']) != entry['sha256']:
            raise ValueError(f"Upstream input checksum mismatch: {entry['path']}")
    if file_hash(corpus_meta) != CORPUS_SHA256:
        raise ValueError('Corpus Parquet SHA-256 differs from the archived input')
    enc = tiktoken.get_encoding('o200k_base')
    order = bedrock + ocean[:tier-len(bedrock)]
    wanted = set(order)
    selected = {}
    # Batch reading bounds temporary memory; selected tier text remains in memory.
    for batch in pq.ParquetFile(corpus_meta).iter_batches(batch_size=2048):
        for row in batch.to_pylist():
            if row['dsid'] in wanted:
                if row['dsid'] in selected:
                    raise ValueError('Duplicate corpus ID')
                selected[row['dsid']] = row
    for dsid, relative in SCAFFOLDS.items():
        text = (benchmark_root/relative).read_text(encoding='utf-8').strip()
        selected[dsid] = {'dsid': dsid, 'text': text, 'source': 'scaffold',
                          'is_noise': False, 'token_len': len(enc.encode(text, disallowed_special=()))}
    if set(selected) != wanted:
        raise ValueError('The archived corpus does not contain all frozen IDs')
    qs = [json.loads(x) for x in (benchmark_root/'questions.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(qs) != 500 or len({q['question_id'] for q in qs}) != 500:
        raise ValueError('Expected the unique 500-question benchmark')
    out.mkdir(parents=True, exist_ok=False)
    (out/'INCOMPLETE').write_text('Materialization has not finished. Do not use this directory.', encoding='utf-8')
    ds2chunks = defaultdict(list)
    cid = 0
    chunk_stream_hash = hashlib.sha256()
    with (out/'chunks.json').open('w',encoding='utf-8',newline='\n') as chunks, \
         (out/'chunks_meta.jsonl').open('w',encoding='utf-8',newline='\n') as metadata, \
         (out/'document_hashes.jsonl').open('w',encoding='utf-8',newline='\n') as hashes:
        chunks.write('[')
        for d in order:
            row = selected[d]
            pieces = chunk_body(row['text'],enc)
            token_count = len(enc.encode(row['text'],disallowed_special=()))
            if token_count != row['token_len']:
                raise ValueError(f'Token-length mismatch for {d}')
            hashes.write(json.dumps({'dsid':d,'text_sha256':hashlib.sha256(row['text'].encode('utf-8')).hexdigest()})+'\n')
            for piece in pieces:
                if cid: chunks.write(', ')
                chunks.write(json.dumps(piece,ensure_ascii=False))
                entry={'chunk_id':cid,'dsid':d,'source':row['source'],'is_noise':bool(row['is_noise'])}
                metadata.write(json.dumps(entry)+'\n')
                ds2chunks[d].append(cid)
                # Length prefix makes concatenation unambiguous.
                raw=piece.encode('utf-8')
                chunk_stream_hash.update(len(raw).to_bytes(8,'big')); chunk_stream_hash.update(raw)
                cid+=1
        chunks.write(']')
    if cid != ref['reference_N_chunks']:
        raise ValueError(f"Chunk count {cid} differs from historical {ref['reference_N_chunks']}")
    with (out/'questions.jsonl').open('w',encoding='utf-8',newline='\n') as stream:
        for q in qs:
            gd=[d for d in (q.get('expected_doc_ids') or []) if d in wanted]
            answer=q.get('gold_answer','')
            row={'id':q['question_id'],'question':q['question'],
                 'answer':[answer] if isinstance(answer,str) else answer,
                 'question_type':q.get('question_type'),'answer_facts':q.get('answer_facts',[]),
                 'gold_dsids':gd,'gold_chunk_ids':[c for d in gd for c in ds2chunks[d]],
                 'has_gold':bool(q.get('expected_doc_ids'))}
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    sources=Counter(selected[d]['source'] for d in order)
    report={'N_docs':len(order),'N_chunks':cid,'manifest_sha256':id_hash(sorted(order)),
            'order_sha256':id_hash(order),'corpus_parquet_sha256':CORPUS_SHA256,
            'chunk_stream_sha256':chunk_stream_hash.hexdigest(),
            'q_with_gold':sum(bool(q.get('expected_doc_ids')) for q in qs),
            'noise_frac':round(sum(bool(selected[d]['is_noise']) for d in order)/len(order),4),
            'source_dist':{s:round(c/len(order),4) for s,c in sources.most_common()},
            'historical_chunk_bytes_verified':False}
    (out/'manifest.json').write_text(json.dumps({'N_docs':len(order),'sha256':report['manifest_sha256'],'dsids':sorted(order)},indent=2),encoding='utf-8')
    (out/'ordered_dsids.txt').write_bytes(('\n'.join(order)+'\n').encode('utf-8'))
    (out/'scale_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    files={p.name:file_hash(p) for p in out.iterdir() if p.is_file() and p.name!='INCOMPLETE'}
    (out/'SHA256SUMS.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
    (out/'INCOMPLETE').unlink()
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--corpus-meta',required=True,type=Path)
    ap.add_argument('--benchmark-root',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--tier',type=int,default=42587)
    args=ap.parse_args()
    print(json.dumps(materialize(args.corpus_meta,args.benchmark_root,args.out,args.tier)))


if __name__=='__main__': main()
