"""Bounded read-only check: compare six blob fingerprints with full/prefix bytes."""
import hashlib,json,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=Path('/home/jon/.cache/vllm-gb10-kv-paging');directories=list(root.glob('*/rank-0/content'));assert len(directories)==1,directories
chosen={}
for f in directories[0].iterdir():
 if f.suffix!='.bin':continue
 group=int(f.name.split('-',1)[0]);chosen.setdefault(group,f)
 if len(chosen)==6:break
rows=[];start=time.time()
for group,f in sorted(chosen.items()):
 expected=f.stem.split('-',1)[1];h=hashlib.sha256();matches=[];size=0
 try:
  with f.open('rb') as stream:
   while chunk:=stream.read(4096):
    h.update(chunk);size+=len(chunk)
    if h.copy().hexdigest()==expected:matches.append(size)
 except FileNotFoundError:continue
 rows.append({'group':group,'file_size':size,'whole_file_matches_group_fingerprint':h.hexdigest()==expected,'matching_prefix_lengths_4k_grid':matches})
result={'start':start,'end':time.time(),'bytes_read':sum(r['file_size'] for r in rows),'scope':'One immutable blob per group, rank0. A whole-file mismatch establishes that fingerprint spans differ from the full file, but does not identify their layout. Prefix scan only checks4KiB boundaries.','rows':rows};(p/'disk-geometry-check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
