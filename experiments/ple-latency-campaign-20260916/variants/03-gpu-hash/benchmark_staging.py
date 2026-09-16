"""Compare complete staging/hash/publication overhead, excluding native gather."""
import json,os,random,runpy,statistics,time
import torch
from torch import nn
ns=runpy.run_path('/check.py')
config=ns['config'];vconfig=ns['vconfig'];Qwen=ns['Qwen4ExpNGramEmbedding'];Connector=ns['PleInProcessConnector']
vconfig.scheduler_config.max_num_batched_tokens=8192;vconfig.scheduler_config.max_num_seqs=32
inputs=torch.arange(8192,dtype=torch.int32,device='cuda');query=torch.zeros(33,dtype=torch.int32,device='cuda');history=torch.arange(64,dtype=torch.int32,device='cuda').reshape(32,2)
connectors={};models=[]
for name,flag in [('cpu_hash','0'),('gpu_hash','1')]:
 os.environ['GB10_PLE_GPU_HASH']=flag
 model=nn.Module();model.layer=Qwen(config,2560,0,8192,32,'layer','layer');models.append(model)
 model.layer._offload_quant_method=ns['Qwen4ExpPLEFp8EmbeddingMethod']()
 con=Connector(vconfig,model,torch.device('cuda:0'),'unused',input_ids_source=inputs,query_start_loc_source=query,ngram_context_source=history)
 for reader in con._readers:reader.gather=lambda *args:None
 connectors[name]=con
rng=random.Random(19);results=[]
for tokens in (4,16,48,128,512,8192):
 reqs=min(32,tokens//4);query[:reqs+1]=torch.arange(reqs+1,device='cuda',dtype=torch.int32)*(tokens//reqs)
 samples={name:[] for name in connectors}
 for iteration in range(120):
  order=list(connectors);rng.shuffle(order)
  for name in order:
   start=time.perf_counter_ns();connectors[name].prepare_forward(reqs,tokens,False);elapsed=(time.perf_counter_ns()-start)/1e3
   if iteration>=20:samples[name].append(elapsed)
  assert torch.equal(connectors['cpu_hash']._hash_state[0][-1][:tokens],connectors['gpu_hash']._hash_state[0][-1][:tokens])
 for name,a in samples.items():
  r=dict(tokens=tokens,rows=tokens*16,requests=reqs,variant=name,median_us=statistics.median(a),p95_us=sorted(a)[94],n=len(a));results.append(r);print(json.dumps(r),flush=True)
for con in connectors.values():con.close()
open('/build/staging-benchmark.json','w').write(json.dumps(results,indent=2)+'\n')
