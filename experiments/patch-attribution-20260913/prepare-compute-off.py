from pathlib import Path
import json,shutil
r=Path('/home/jon/dual-spark-inference-kv-paging');e=r/'experiments/patch-attribution-20260913';e.mkdir(exist_ok=True)
shutil.copy2(r/'deploy_config.json',e/'config.before.json')
c=json.loads((r/'deploy_config.json').read_text())
for k in ['bf16_kernels','moe_down_tensor','moe_up_tensor','moe_sparse_activation','draft_head_gemm','ple_native_hash','ple_mapped_transport']:
 c['optimizations'][k]=False
c['optimizations']['ple_gather_threads']=0;c['optimizations']['ple_row_cache_mb']=0
c['enable_roce_allreduce']=False
c['runtime_overrides']={};c['runtime_override_sha256']={}
(r/'deploy_config.json').write_text(json.dumps(c,indent=2)+'\n')
(e/'config.compute-off.json').write_text(json.dumps(c,indent=2)+'\n')
