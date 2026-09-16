from pathlib import Path
import hashlib,json
p=Path(__file__).resolve().parent;old=p.parent/'targeted-determinism-20260913/targeted_kernels.dense.py';s=old.read_text()
s=s.replace('''    grid = (min(sms, triton.cdiv(m, 128) * triton.cdiv(n, 128)),)''','''    # Shape-specific tuning, constant across every batch size. K remains64.
    # Only the traced HC-down and GDN-ba projections opt into this plan.
    config = (dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
                   GROUP_SIZE_M=8, num_stages=3, num_warps=4)
              if (n, k) in {(336, 10240), (48, 2560)} else GEMM_CONFIG)
    grid = (min(sms, triton.cdiv(m, config["BLOCK_SIZE_M"]) * triton.cdiv(n, config["BLOCK_SIZE_N"])),)''',1)
s=s.replace('''        **GEMM_CONFIG)''','''        **config)''',1)
f=p/'targeted_kernels.tuned.py';f.write_text(s);compile(s,str(f),'exec')
for rank in [0,1]:
 cfg=json.loads((p/f'candidate-content-r{rank}.json').read_text());key='model_executor/determinism/gb10_targeted.py';cfg['runtime_overrides'][key]=str(f);cfg['runtime_override_sha256'][key]=hashlib.sha256(f.read_bytes()).hexdigest();(p/f'candidate-tuned-r{rank}.json').write_text(json.dumps(cfg,indent=2)+'\n')
print('Candidate D prepared only; fixed64x64 tiles for traced HC-down/GDN-ba shapes, K64 for all batches.')
