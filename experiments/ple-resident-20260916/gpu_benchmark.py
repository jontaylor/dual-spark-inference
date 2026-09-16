"""Paired complete PLE prepare->GPU byte consumption on identical CUDA inputs."""
import ctypes, json, os, random, runpy, statistics, time
import torch
from torch import nn
ns=runpy.run_path('/check.py')
config=ns['config'];Qwen=ns['Qwen4ExpNGramEmbedding'];Connector=ns['PleInProcessConnector'];vconfig=ns['vconfig'];table=ns['table']
vconfig.scheduler_config.max_num_batched_tokens=128;vconfig.scheduler_config.max_num_seqs=32
os.environ['GB10_PLE_ROW_CACHE_MB']='0'
inputs=torch.arange(128,dtype=torch.int32,device='cuda');query=torch.zeros(33,dtype=torch.int32,device='cuda');history=torch.arange(64,dtype=torch.int32,device='cuda').reshape(32,2)
model=nn.Module();model.layer=Qwen(config,2560,0,128,32,'layer','layer');model.layer._offload_quant_method=ns['Qwen4ExpPLEFp8EmbeddingMethod']()
con=Connector(vconfig,model,torch.device('cuda:0'),'unused',input_ids_source=inputs,query_start_loc_source=query,ngram_context_source=history)
r=con._readers[0];r.lib.gb10_ple_reader_resident_mode.argtypes=[ctypes.c_void_p,ctypes.c_int]
legacy=ctypes.CDLL('/opt/gb10/libple_gather.so');legacy.gb10_ple_gather.argtypes=[ctypes.c_void_p,ctypes.c_int64,ctypes.c_int64,ctypes.c_void_p,ctypes.c_int64,ctypes.c_void_p,ctypes.c_int]
original_submit=r.submit;original_complete=r.complete
results=[];rng=random.Random(418)
for reqs in [1,12,24,32]:
 tokens=reqs*4;query[:reqs+1]=torch.arange(reqs+1,dtype=torch.int32,device='cuda')*4
 con.signal_dummy_outputs(tokens);torch.cuda.synchronize()
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):actual=model.layer(torch.empty(0,device='cuda'),inputs[:tokens]).view(torch.uint8).clone()
 samples={name:[] for name in ['current_sqpoll','batched_prefetch','legacy16_no_ipc','legacy1_no_ipc']}
 for step in range(70):
  inputs.copy_((torch.arange(128,device='cuda',dtype=torch.int32)+step)%900)
  order=list(samples);rng.shuffle(order)
  for name in order:
   if name.startswith('legacy'):
    threads=16 if name.startswith('legacy16') else 1
    def submit(ids,out,timeout,threads=threads):
     code=legacy.gb10_ple_gather(table.data_ptr(),table.shape[0],table.shape[1],ids.data_ptr(),ids.numel(),out.data_ptr(),threads)
     assert code==0
    r.submit=submit;r.complete=lambda:None
   else:
    r.submit=original_submit;r.complete=original_complete
    assert r.lib.gb10_ple_reader_resident_mode(r.handle,int(name=='batched_prefetch'))==0
   start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
   begin=time.perf_counter_ns();start.record()
   con.prepare_forward(reqs,tokens,False,defer_completion=True);graph.replay();end.record();con.finish_forward();end.synchronize()
   wall=(time.perf_counter_ns()-begin)/1e6;gpu=start.elapsed_time(end)
   expected=table.index_select(0,con._hash_state[0][-1][:tokens].flatten()).reshape(tokens,2560)
   assert torch.equal(actual.cpu(),expected),name
   con.release_outputs()
   if step>=10:samples[name].append({'wall_ms':wall,'gpu_ms':gpu})
 for name,values in samples.items():
  row={'requests':reqs,'rows':tokens*16,'variant':name,'n':len(values)}
  for metric in ['wall_ms','gpu_ms']:row[metric]={'median':statistics.median(v[metric] for v in values),'p95':sorted(v[metric] for v in values)[56]}
  results.append(row);print(json.dumps(row),flush=True)
con.close()
open('/results/gpu-benchmark.json','w').write(json.dumps(results,indent=2)+'\n')
