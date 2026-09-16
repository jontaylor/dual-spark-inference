"""Validate trace attribution, physical slots, TP agreement and seed replay."""
import json
from pathlib import Path
import sys
import torch

head, worker, original_api, seeded_api = map(Path, sys.argv[1:])
load = lambda p: [torch.load(f, weights_only=False) for f in sorted(p.glob('*.pt'))]
rs, rt = load(head), load(worker)
assert len(rs) == len(rt)
tp = []
for a,b in zip(rs,rt):
    tp.append({'step': a['step'],
        'request_ids': a['batch']['req_ids'] == b['batch']['req_ids'],
        'inputs': torch.equal(a['batch']['input_ids'], b['batch']['input_ids']),
        'positions': torch.equal(a['batch']['positions'], b['batch']['positions']),
        'logits': torch.equal(a['raw_logits'], b['raw_logits']),
        'hidden': torch.equal(a['sample_hidden_states'], b['sample_hidden_states'])})

slot_fail, overlaps, mamba_fail = [], [], []
slot_checks = 0
for t in rs:
    b=t['batch']; n=b['num_reqs']; mamba_expected=[]
    for g,size in enumerate([1600,1600,1600,1600,4,1600]):
        blocks=t['block_tables'][g]; used=[]
        for i in range(n):
            start,end=map(int,b['query_start_loc_np'][i:i+2])
            positions=b['positions'][start:end].long()
            logical=positions//size
            if g==4:
                # CircularBufferSpec deliberately disables generic slot mapping.
                # Its backend indexes its dedicated per-request ring itself.
                logical=logical%int(t['num_blocks'][g,i])
                expected=torch.full_like(positions,-1)
            else:
                expected=blocks[i,logical].long()*size+positions%size
            actual=t['slot_mappings'][g,start:end].long()
            slot_checks+=1
            if not torch.equal(expected,actual):slot_fail.append([t['step'],g,i])
            writable=set(blocks[i,logical].tolist())
            for j,prior in enumerate(used):
                if prior&writable:overlaps.append([t['step'],g,j,i,sorted(prior&writable)])
            used.append(writable)
        if g<4:
            mamba_expected.append([int(blocks[i,int(t['_mamba_state_idx_gpu'][b['idx_mapping'][i]])]) for i in range(n)])
    for name,md in t['attention_metadata'].items():
        if not isinstance(md,dict):continue
        inds=md.get('non_spec_state_indices_tensor')
        if isinstance(inds,torch.Tensor) and inds[:n].tolist() not in mamba_expected:
            mamba_fail.append([t['step'],name,inds[:n].tolist()])

api={}
for folder in [original_api,seeded_api]:
    for r in json.loads((folder/'results.json').read_text()):api[r['request_id']]=r
api_fail=[]; api_count=0; max_error=0
for t in rs:
    b=t['batch']
    for i,req in enumerate(b['req_ids']):
        key=next(k for k in api if req.startswith('chatcmpl-'+k+'-'))
        r=api[key]
        pred=int(b['positions'][b['logits_indices'][i]])-int(t['prompt_len'][i])+2
        token=r['token_ids'][pred-1]
        if token != int(t['sampler_output']['sampled_token_ids'][i,0]):api_fail.append([t['step'],req,pred,'token'])
        logits=t['raw_logits'][i].float()
        expected=float(logits[token]-torch.logsumexp(logits,dim=0))
        error=abs(expected-r['logprobs'][pred-1]['logprob'])
        max_error=max(max_error,error)
        if error>1e-4:api_fail.append([t['step'],req,pred,'logprob',error])
        api_count+=1

def index(seed):
    out={}
    for t in rs:
        if ('captured-seed' in t['phase'])!=seed:continue
        b=t['batch'];phase=t['phase'].replace('captured-seed','captured')
        for i in range(b['num_reqs']):
            pred=int(b['positions'][b['logits_indices'][i]])-int(t['prompt_len'][i])+2
            name=b['req_ids'][i].split('serial-')[-1].rsplit('-',1)[0] if 'serial-' in b['req_ids'][i] else str(i)
            out[(phase,name,pred)]=(t['raw_logits'][i],t['sample_hidden_states'][i])
    return out
x,y=index(False),index(True)
summary={
    'tp_steps':len(tp), 'tp_mismatches':[c for c in tp if not all(v for k,v in c.items() if k!='step')],
    'slot_mapping_checks':slot_checks, 'slot_mapping_failures':slot_fail,
    'writable_block_aliases':overlaps, 'mamba_metadata_failures':mamba_fail,
    'api_rows':api_count, 'api_failures':api_fail, 'max_api_logprob_abs_error':max_error,
    'seed_rows':len(x), 'seed_matching_keys':x.keys()==y.keys(),
    'seed_mismatches':[str(k) for k in x if k not in y or not all(torch.equal(a,b) for a,b in zip(x[k],y[k]))],
}
(head/'checks.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
