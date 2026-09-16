"""GPU mapped-buffer integration with a small exact synthetic packed table."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from torch import nn
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpNGramEmbedding, Qwen4ExpPLEFp8EmbeddingMethod
from vllm.v1.ple_offload.in_process import PleInProcessConnector

config=NS(ngram_size=3,heads_per_ngram=8,eos_token_id=999,vocab_size=1000,seed=1234,ngram_vocab_size_base=101,make_ngram_vocab_size_divisible_by=128)
size,offset,rows=Qwen4ExpNGramEmbedding._make_vocab_layout(ngram_vocab_size_base=101,ngram_heads=16,ple_dense_layer_id=0)
rows=((rows+127)//128)*128
root=Path('/tmp/ple-test-table');root.mkdir(exist_ok=True)
torch.manual_seed(7)
table=torch.randint(0,256,(rows,160),dtype=torch.uint8)
(root/'layer.ngram_embedding.packed_u8').write_bytes(bytes(table.flatten().tolist()))
(root/'layer.ngram_embedding.packed_u8.json').write_text(json.dumps({'row_format':'fp8_e4m3','total_rows':rows,'row_width':160}))
vconfig=NS(parallel_config=NS(nnodes=2,tensor_parallel_size=2,data_parallel_size=1),scheduler_config=NS(max_num_batched_tokens=16,max_num_seqs=4),model_config=NS(hf_text_config=config,dtype=torch.bfloat16))
for source_device in ['cpu','cuda']:
 model=nn.Module();model.layer=Qwen4ExpNGramEmbedding(config,2560,0,16,4,'layer','layer')
 # Deliberately different checkpoint hash parameters must be respected.
 multipliers=torch.tensor([11,23,37],dtype=torch.int64)
 model.layer._offload_quant_method=Qwen4ExpPLEFp8EmbeddingMethod()
 retained=model.layer.load_weights(iter([('layer_multipliers',multipliers),('ngram_heads_vocab_sizes',torch.tensor(size)),('ngram_heads_offsets',torch.tensor(offset)),('ngram_embedding.weight_scale',torch.tensor(1.0))]))
 assert retained=={'layer_multipliers','ngram_heads_vocab_sizes','ngram_heads_offsets','ngram_embedding.weight_scale'}
 inputs=torch.arange(16,dtype=torch.int32,device=source_device)
 query=torch.tensor([0,4,8,12,16],dtype=torch.int32,device=source_device)
 history=torch.tensor([[1,2],[999,4],[5,6],[7,8]],dtype=torch.int32,device=source_device)
 connector=PleInProcessConnector(vconfig,model,torch.device('cuda:0'),'unused',input_ids_source=inputs,query_start_loc_source=query,ngram_context_source=history)
 for step in range(3):
  connector.prepare_forward(4,16,False)
  expected=table.index_select(0,connector._hash_state[0][-1].flatten()).reshape(16,2560)
  actual=model.layer(torch.empty(0,device='cuda'),inputs).view(torch.uint8)
  assert torch.equal(actual.cpu(),expected)
  connector.release_outputs()
  inputs.add_(1)
 connector.signal_dummy_outputs(16)
 torch.cuda.synchronize()
 assert torch.count_nonzero(model.layer._gpu_output_buffer.view(torch.uint8)).item()==0
 connector.close();connector.close()
 assert torch.equal(connector._hash_state[0][3][0],multipliers)
 print('PASS',source_device,'staging, native batch reads, mapped GPU bytes, buffer reuse, checkpoint hash, dummy output',flush=True)
