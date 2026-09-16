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


import importlib.util,sys
spec=importlib.util.spec_from_file_location('vllm.model_executor.determinism.gb10_targeted','/e/targeted_kernels.py')
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)

import importlib.util, types
spec=importlib.util.spec_from_file_location('vllm.models.qwen4_exp.nvidia.hc_candidate','/e/hyperconnection.fixed.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
obj=module.GatedResidual.__new__(module.GatedResidual);torch.nn.Module.__init__(obj)
obj.config=types.SimpleNamespace(rms_norm_eps=cfg['rms_norm_eps'])
obj.hc_count=hc;obj.lora_rank=lowrank;obj.pad_size=pad;obj.use_combine=True
for name,w in [('hc_norm',wn),('input_mix_weight_down_block_inject',wd),('input_mix_weight_up',wu)]:
 holder=torch.nn.Module();holder.weight=torch.nn.Parameter(w,requires_grad=False);setattr(obj,name,holder)

with torch.inference_mode():
    records=[]
    # Four independent input rows, evaluated alone versus in one M=4 call.
    decode_x=xs[:4].contiguous()
    calls=[]
    original=module.fixed_bf16_matmul
    def tracked(a,b,bias=None):
        calls.append([int(a.shape[0]),int(a.shape[1]),int(b.shape[1])])
        return original(a,b,bias)
    module.fixed_bf16_matmul=tracked
    one=[obj.mix(decode_x[i:i+1]) for i in range(4)]
    calls.clear()
    batch=obj.mix(decode_x)
    batched_calls=list(calls)
    assert len(batched_calls)==2 and all(x[0]==4 for x in batched_calls)
    for j,name in [(1,'block_input'),(2,'injection')]:
        reference=torch.cat([x[j] for x in one]);equal=torch.equal(reference,batch[j])
        records.append({'stage':'HC M1 vs M4','tensor':name,'equal':equal,'batched_gemm_calls':batched_calls})
        assert equal
    module.fixed_bf16_matmul=original
    a=obj.mix(xs);b=obj.mix(xm)
    for i in [1,2,3]:
        start,end=map(int,m['batch']['query_start_loc_np'][i:i+2])
        for j,name in [(1,'block_input'),(2,'injection')]:
            equal=torch.equal(a[j],b[j][start:end]);assert equal
            records.append({'stage':'HC M330 vs M991','row':i,'tensor':name,'equal':equal})
    for projection in ['in_proj_qkvz','in_proj_ba']:
        weights=torch.load('/weights/language_model.model.layers.0.linear_attn.'+projection+'.pt',weights_only=False)
        w=weights['weight'].cuda();bias=weights['bias'];assert bias is None
        for kind,op in [('stock',lambda x:torch.nn.functional.linear(x,w)),('fixed',lambda x:mod.fixed_bf16_matmul(x,w.t()))]:
            u=op(a[1]);v=op(b[1])[1:331]
            d=u.float()-v.float();records.append({'stage':projection,'method':kind,'equal':torch.equal(u,v),'changed':int((d!=0).sum()),'max_abs':float(d.abs().max()),'weight_dtype':str(w.dtype),'weight_shape':list(w.shape)})

    for layer in [0,2]:
        for projection in ['in_proj_qkvz','in_proj_ba','out_proj']:
            weights=torch.load(f'/weights/language_model.model.layers.{layer}.linear_attn.'+projection+'.pt',weights_only=False)
            w=weights['weight'].cuda()
            if projection=='out_proj':
                entry=next(e for e in s['full_ops'] if e['name']==f'language_model.model.layers.{layer}.linear_attn.out_proj' and e['event']=='input')
                x=entry['values']['args'][0][:4].cuda()
            else:x=a[1][:4]
            for kind,op in [('stock',lambda x:torch.nn.functional.linear(x,w)),('fixed',lambda x:mod.fixed_bf16_matmul(x,w.t()))]:
                u=torch.cat([op(x[i:i+1]) for i in range(4)]);v=op(x)
                d=u.float()-v.float();records.append({'stage':f'layer{layer}.'+projection+' M1 vs M4','method':kind,'equal':torch.equal(u,v),'changed':int((d!=0).sum()),'max_abs':float(d.abs().max())})
    from vllm.third_party.flash_linear_attention.ops import layernorm_guard as lg
    for layer in [0,2]:
        entry=next(e for e in s['full_ops'] if e['name']==f'language_model.model.layers.{layer}.linear_attn.out_proj' and e['event']=='input')
        xx=entry['values']['args'][0].reshape(-1,128);zz=xx.roll(7,0);ww=weight(f'model.language_model.layers.{layer}.linear_attn.norm.weight').cuda()
        rows_per_token=xx.shape[0]//330
        xx=xx[:64*rows_per_token].cuda();zz=zz[:64*rows_per_token].cuda()
        for kind in ['stock','fixed_rows']:
            old=lg.calc_rows_per_block
            if kind=='fixed_rows':lg.calc_rows_per_block=lambda M,device:1
            try:
                u=torch.cat([lg.rmsnorm_fn(xx[i:i+rows_per_token],ww,None,z=zz[i:i+rows_per_token],eps=cfg['rms_norm_eps'],norm_before_gate=True) for i in range(0,len(xx),rows_per_token)])
                v=lg.rmsnorm_fn(xx,ww,None,z=zz,eps=cfg['rms_norm_eps'],norm_before_gate=True)
                d=u.float()-v.float();records.append({'stage':f'layer{layer}.RMSNormGated','method':kind,'equal':torch.equal(u,v),'changed':int((d!=0).sum()),'max_abs':float(d.abs().max())})
            finally:lg.calc_rows_per_block=old
    out={'fixed_config':mod.GEMM_CONFIG,'records':records}
    print(json.dumps(out,indent=2))
    Path('/e/projection-replay.json').write_text(json.dumps(out,indent=2))
