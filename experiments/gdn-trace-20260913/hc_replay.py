"""Replay layer-0 HyperConnection from checkpoint weights, no model service."""
import os
import sys
MODE=sys.argv[1] if len(sys.argv)>1 else 'baseline'
if MODE in ('batch-invariant','patched-batch-invariant'):
    os.environ['VLLM_BATCH_INVARIANT']='1'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
    os.environ['CUBLASLT_WORKSPACE_SIZE']='1'
import json
from pathlib import Path
import torch
from safetensors import safe_open
if MODE=='patched-batch-invariant':
    import importlib.util
    name='vllm.model_executor.determinism.batch_invariant'
    spec=importlib.util.spec_from_file_location(name,'/tmp/batch_invariant.sm12x.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    module.init_batch_invariance()
elif MODE=='batch-invariant':
    from vllm.model_executor.determinism.batch_invariant import init_batch_invariance
    init_batch_invariance()
from vllm.models.qwen4_exp.nvidia.ops.hc import grouped_gemma_rmsnorm,hc_silu,hc_gate_mix

torch.backends.cuda.matmul.fp32_precision='ieee'
if MODE=='no-bf16-reduction':
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
root=Path('/model-store/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47')
cfg=json.loads((root/'config.json').read_text())['text_config']
index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
def weight(name):
    with safe_open(root/index[name],framework='pt',device='cpu') as f:
        return f.get_tensor(name)

prefix='model.language_model.layers.0.attn_hyper_connection.'
wn=weight(prefix+'hc_norm.weight').cuda()
wd=weight(prefix+'input_mix_weight_down.weight')
wi=weight(prefix+'block_inject_weight.weight')
wu=weight(prefix+'input_mix_weight_up.weight').cuda()
lowrank=wd.shape[0]; hc=cfg['hc_count']
pad=(-(lowrank+hc))%16
wd=torch.cat([wd,wi,torch.zeros(pad,wd.shape[1],dtype=wd.dtype)]).cuda()
we=weight('model.language_model.embed_tokens.weight')
s=torch.load('/tmp/vllm-row-trace/279-0001.pt',weights_only=False)
m=torch.load('/tmp/vllm-row-trace/279-0017.pt',weights_only=False)
def expected(r):
    return next(e for e in r['full_ops'] if e['name']=='language_model.model.layers.0.linear_attn' and e['event']=='input')['values']['kwargs']['hidden_states']
def inputs(r):
    ids=r['batch']['input_ids'][:r['batch']['num_tokens']].long()
    return we[ids].repeat(1,hc).cuda()
xs,xm=inputs(s),inputs(m)
del we

def run(x):
    xn=grouped_gemma_rmsnorm(x,wn,cfg['rms_norm_eps'],hc)
    if MODE.startswith('persistent'):
        from vllm.model_executor.determinism.batch_invariant import matmul_persistent
        down=matmul_persistent(xn,wd.t())
    elif MODE in ('fp32-down','fp32-both'):
        down=torch.nn.functional.linear(xn.float(),wd.float()).to(xn.dtype)
    else:down=torch.nn.functional.linear(xn,wd)
    silu=hc_silu(down[:,:lowrank],hc)
    if MODE=='persistent-both':gate=matmul_persistent(silu,wu.t())
    elif MODE=='fp32-both':gate=torch.nn.functional.linear(silu.float(),wu.float()).to(xn.dtype)
    else:gate=torch.nn.functional.linear(silu,wu)
    block=hc_gate_mix(xn,gate,hc)
    return {'normalized':xn.cpu(),'down':down[:,:lowrank+hc].cpu(),'silu':silu.cpu(),'gate':gate.cpu(),'block_input':block.cpu()}

a=run(xs);b=run(xm);again=run(xm)
out={'mode':MODE,'weight_dtypes':[str(w.dtype) for w in [wn,wd,wu]],
     'bf16_reduced_precision':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
     'repeat_equal':{k:torch.equal(b[k],again[k]) for k in b},
     'serving_serial_equal':torch.equal(a['block_input'],expected(s)),
     'serving_mixed_equal':torch.equal(b['block_input'],expected(m)),
     'comparisons':[]}
for i in [1,2,3]:
    start,end=map(int,m['batch']['query_start_loc_np'][i:i+2])
    for k in a:
        x=a[k].float();y=b[k][start:end].float();d=y-x
        out['comparisons'].append({'row':i,'operation':k,'equal':torch.equal(x,y),
            'max_abs':float(d.abs().max()),'changed_elements':int((d!=0).sum()),
            'changed_tokens':int((d!=0).any(dim=-1).sum()),
            'relative_l2':float(d.norm()/x.norm().clamp_min(1e-20))})
Path('/tmp/hc-replay-'+MODE+'.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))

if MODE=='profile':
    normalized_s=grouped_gemma_rmsnorm(xs,wn,cfg['rms_norm_eps'],hc)
    normalized_m=grouped_gemma_rmsnorm(xm,wn,cfg['rms_norm_eps'],hc)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
        with torch.profiler.record_function('HC_DOWN_330'):
            torch.nn.functional.linear(normalized_s,wd)
        with torch.profiler.record_function('HC_DOWN_991'):
            torch.nn.functional.linear(normalized_m,wd)
        torch.cuda.synchronize()
    prof.export_chrome_trace('/tmp/hc-kernels.json')
    kernels=[{'name':e.name,'device':str(e.device_type),'duration_us':e.device_time_total} for e in prof.events() if 'CUDA' in str(e.device_type)]
    Path('/tmp/hc-kernel-names.json').write_text(json.dumps(kernels,indent=2))
    print('KERNELS',json.dumps(kernels,indent=2))
