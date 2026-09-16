import json
from pathlib import Path
import sys
import torch
import numpy as np

root = Path(sys.argv[1])
records = []
failures = []
for file in sorted(root.glob('*.pt')):
    t = torch.load(file, weights_only=False)
    b = t['batch']; n = b['num_reqs']
    if 'raw_logits' not in t:
        failures.append([file.name, 'missing logits']); continue
    for i, req in enumerate(b['req_ids']):
        state_idx = int(b['idx_mapping_np'][i])
        pos = int(b['positions'][b['logits_indices'][i]])
        prompt_len = int(t['prompt_len'][i])
        pred = pos - prompt_len + 2
        a, z = map(int, b['query_start_loc_np'][i:i+2])
        positions = b['positions'][a:z].long()
        input_ids = b['input_ids'][a:z]
        saved_tokens = t['state_token_ids'][i]
        checks = {
            'mapping_cpu_gpu': int(b['idx_mapping'][i]) == state_idx,
            'mapping_request': t['request_id_to_index'][req] == state_idx,
            'mapping_reverse': t['index_to_request_id'][state_idx] == req,
            'sampling_request': t['sampling_batch']['req_ids'][i] == req,
            'computed_cpu_gpu': int(t['computed_gpu'][i]) == int(b['num_computed_tokens_np'][i]),
            'position_start': int(positions[0]) == int(t['computed_gpu'][i]),
            'positions_contiguous': torch.equal(positions, torch.arange(int(positions[0]), int(positions[0])+len(positions))),
            'position_seq_len': pos + 1 == int(b['seq_lens'][i]),
            'input_matches_request_state': torch.equal(input_ids.long(), saved_tokens[positions].long()),
            'temperature_cpu_zero': float(t['temperature_cpu'][i]) == 0,
            'temperature_gpu_zero': float(t['temperature_gpu'][i]) == 0,
        }
        logits = t['raw_logits'][i].float()
        sampled = int(t['sampler_output']['sampled_token_ids'][i,0])
        checks['sample_is_raw_max'] = float(logits[sampled]) == float(logits.max())
        failures.extend([file.name, req, pred, k] for k,v in checks.items() if not v)
        values, tokens = logits.topk(5)
        tables = [table[i, :int(t['num_blocks'][g,i])].tolist()
                  for g,table in enumerate(t['block_tables'])]
        records.append({'file':file.name, 'phase':t['phase'], 'request':req,
            'prediction':pred, 'row':i, 'state_index':state_idx,
            'num_reqs':n, 'padded_tokens':b['num_tokens_after_padding'],
            'positions':[int(positions[0]),pos], 'sampled':sampled,
            'top_tokens':tokens.tolist(), 'top_logits':values.tolist(),
            'look_minus_investigate':float(logits[1353]-logits[18730]),
            'block_tables':tables, 'checks':checks,
            'hidden':t['sample_hidden_states'][i].float(), 'logits':logits,
            'prefix':saved_tokens[:pos+1].long()})

comparisons = []
for phase in sorted(set(r['phase'].rsplit('-', 1)[0] for r in records)):
    pass
for r in records:
    if '-parallel-' not in r['request']:
        continue
    group = 'captured-seed' if 'rowtrace-captured-seed-' in r['request'] else 'captured'
    serial = next((s for s in records if f'rowtrace-{group}-serial-before-0-' in s['request'] and s['prediction']==r['prediction']),None)
    if serial is None: continue
    same_prefix = torch.equal(serial['prefix'],r['prefix'])
    hd = r['hidden']-serial['hidden']; ld=r['logits']-serial['logits']
    comparisons.append({'request':r['request'], 'prediction':r['prediction'],
        'row':r['row'], 'same_prefix':same_prefix,
        'hidden_max_abs':float(hd.abs().max()),
        'hidden_rel_l2':float(hd.norm()/serial['hidden'].norm()),
        'logit_max_abs':float(ld.abs().max()),
        'look_minus_investigate_serial':serial['look_minus_investigate'],
        'look_minus_investigate_concurrent':r['look_minus_investigate'],
        'serial_sampled':serial['sampled'], 'concurrent_sampled':r['sampled']})

serializable = [{k:v for k,v in r.items() if k not in ('hidden','logits','prefix')} for r in records]
out={'files':len(list(root.glob('*.pt'))),'rows':len(records),'failures':failures,
     'records':serializable,'comparisons':comparisons}
(root/'analysis.json').write_text(json.dumps(out,indent=2))
print('files',out['files'],'rows',len(records),'failures',failures)
for r in comparisons:
    if r['prediction']<=3:
        print(json.dumps(r))
