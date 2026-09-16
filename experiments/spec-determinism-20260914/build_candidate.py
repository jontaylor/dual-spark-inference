import json,hashlib,py_compile
from pathlib import Path
root=Path(__file__).resolve().parents[2];e=Path(__file__).resolve().parent;old=root/'experiments/targeted-determinism-20260913'
s=(old/'gdn.execution.py').read_text();needle='from vllm.third_party.flash_linear_attention.ops.chunk import l2norm_fwd';s=s.replace(needle,'from vllm.third_party.flash_linear_attention.ops.gb10_spec_recurrent import fused_sigmoid_gating_delta_rule_update\n'+needle);(e/'gdn.spec.py').write_text(s)
s=(old/'scheduler.fixed.py').read_text();s=s.replace('if self.use_eagle:\n                raise ValueError("Targeted GDN invariance does not support speculation")','if self.use_eagle and (speculative_config.method != "mtp" or self.num_spec_tokens > 3):\n                raise ValueError("Targeted GDN invariance supports only MTP with up to 3 draft tokens")');(e/'scheduler.spec.py').write_text(s)
for rank in [0,1]:
 cfg=json.loads((old/f'candidate-final-r{rank}.json').read_text());cfg['mtp_tokens']=3
 updates={'model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py':e/'gdn.spec.py','v1/core/sched/scheduler.py':e/'scheduler.spec.py','third_party/flash_linear_attention/ops/gb10_spec_recurrent.py':e/'fused_sigmoid.fixed.py'}
 for key,p in updates.items():
  py_compile.compile(str(p),doraise=True);cfg['runtime_overrides'][key]=str(p);cfg['runtime_override_sha256'][key]=hashlib.sha256(p.read_bytes()).hexdigest()
 (e/f'candidate-r{rank}.json').write_text(json.dumps(cfg,indent=2)+'\n')
print('Candidate assembled; live configs unchanged.')
