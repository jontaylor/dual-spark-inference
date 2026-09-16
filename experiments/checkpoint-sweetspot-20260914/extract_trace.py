"""Match append-only follow-ups and estimate interval replay, without inference."""
import hashlib
import json
import statistics
from pathlib import Path

root=Path('/home/jon/artifact-agent-orchestration/.artifact-agent/baseline-comparison')
def signature(m):
    # The harness reserializes JSON arguments. This checks semantic identity of
    # the appended output, not generated-token identity (which is unavailable).
    calls=[]
    for call in m.get('tool_calls') or []:
        c=dict(call); c.pop('id',None)
        f=dict(c.get('function',{})); a=f.get('arguments')
        if isinstance(a,str):
            try:f['arguments']=json.loads(a)
            except json.JSONDecodeError:pass
        c['function']=f;calls.append(c)
    return {'role':m.get('role'), 'content':m.get('content') or '',
            'reasoning':m.get('reasoning_content',m.get('reasoning')) or '',
            'tool_calls':calls}

result={}
for name in ['20260914T113242Z-temperature-warm-repeat','20260914T123549Z-temperature-overlap']:
    rows=[]
    for arm in sorted((root/name).glob('temperature-*')):
        if not arm.is_dir():continue
        previous={}
        for inp in sorted(arm.glob('model-input-*.json')):
            output=inp.with_name(inp.name.replace('model-input-','model-output-'))
            if not output.exists():continue
            d=json.loads(inp.read_text());o=json.loads(output.read_text())
            if not isinstance(o.get('usage'),dict):continue
            messages=d.get('messages',[])
            h=hashlib.sha256();match=None
            # Include every earlier wire-message prefix, not merely the previous
            # request number: concurrent workers can interleave different chains.
            for message in messages:
                raw=json.dumps(message,sort_keys=True,separators=(',',':')).encode()
                h.update(len(raw).to_bytes(8,'little'));h.update(raw)
                old=previous.get(h.digest())
                if old is not None:match=old
            u=o['usage'];prompt=u['prompt_tokens'];generated=u['completion_tokens']
            cached=u.get('prompt_tokens_details',{}).get('cached_tokens',0)
            current={'file':inp.name,'prompt':prompt,'generated':generated,
                     'message_count':len(messages),'created':o.get('created'),
                     'assistant':signature(o['choices'][0]['message'])}
            if match is not None and match['file']!=inp.name:
                end=match['prompt']+match['generated']-1
                # Token-count estimates are invalid when rendering changed or
                # history was shortened despite matching earlier wire messages.
                if prompt>=end:
                    row={'arm':arm.name,'file':inp.name,'previous':match['file'],
                        'created':o.get('created'),'previous_created':match['created'],
                        'previous_output_semantically_matches_history':(
                            len(messages)>match['message_count'] and
                            signature(messages[match['message_count']])==match['assistant']),
                        'prompt':prompt,'cached':cached,'uncached':prompt-cached,
                        'previous_prompt':match['prompt'],'previous_generated':match['generated'],
                        'estimated_previous_computed':end,'estimated_new_tokens':prompt-end}
                    row['older_prompt_replay_floor']=max(0,match['prompt']-cached)
                    row['previous_output_reused_estimate']=max(0,min(end,cached)-match['prompt'])
                    row['candidates']={str(interval):{
                        'estimated_replay':end%interval,
                        'estimated_uncached':prompt-end+end%interval,
                        'decode_checkpoints':max(0,end//interval-match['prompt']//interval)}
                        for interval in [64,128,256,512,1024,1920]}
                    rows.append(row)
            previous[h.digest()]=current
    summary={'matched_followups':len(rows),'scope':'Token-count estimate on exact append-only wire-message prefixes. Not a tokenizer proof or measured candidate cache result.'}
    if rows:
        for field in ['uncached','estimated_new_tokens','previous_generated']:
            summary['mean_'+field]=statistics.mean(r[field] for r in rows)
        summary['candidates']={str(i):{field:statistics.mean(r['candidates'][str(i)][field] for r in rows)
            for field in ['estimated_replay','estimated_uncached','decode_checkpoints']} for i in [64,128,256,512,1024,1920]}
    result[name]={'summary':summary,'rows':rows}
print(json.dumps(result,indent=2))
