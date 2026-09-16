import hashlib,json,os
from pathlib import Path
root=Path(__file__).resolve().parent
seen={};records=[];saved=0
for p in sorted(root.rglob('*.pt')):
 st=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 key=(st.st_size,h.hexdigest())
 original=seen.get(key)
 if original is None:seen[key]=p;continue
 if original.stat().st_ino==st.st_ino:continue
 temp=p.with_name(p.name+'.dedup-link');os.link(original,temp);os.replace(temp,p)
 records.append({'path':str(p.relative_to(root)),'identical_to':str(original.relative_to(root)),'sha256':h.hexdigest(),'bytes':st.st_size});saved+=st.st_size
(root/'evidence-deduplication.json').write_text(json.dumps({'bytes_reclaimed':saved,'links':records},indent=2))
print('Preserved',len(records),'duplicate paths; reclaimed GiB',saved/1024**3)
