import importlib.util,json,torch
from pathlib import Path
spec=importlib.util.spec_from_file_location('qsa_check','/e/qsa_ops.original.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.manual_seed(19);q=torch.randn(1,4,128,device='cuda',dtype=torch.bfloat16);k=torch.randn(2,400,1,128,device='cuda',dtype=torch.bfloat16)
bt=torch.tensor([[0,1]],device='cuda',dtype=torch.int32);lens=torch.tensor([1930],device='cuda',dtype=torch.int32)
rows=[];refs=None
for M in [1,4,330,991]:
 args=(q.repeat(M,1,1),k,bt,torch.zeros(M,device='cuda',dtype=torch.int32),torch.full((M,),1929,device='cuda',dtype=torch.int32),lens)
 scores,visible=m.qsa_mqa_paged(*args,compress_ratio=4)
 indices=m.qsa_select_paged_tokens(*args,token_topk=2048,compress_ratio=4)
 if refs is None:refs=(scores[0].clone(),indices[0].clone())
 row={'M':M,'scores_equal':bool(torch.all(scores==refs[0])),'indices_equal':bool(torch.all(indices==refs[1]))};rows.append(row);assert row['scores_equal'] and row['indices_equal'],row
Path('/e/qsa-index-check.json').write_text(json.dumps(rows,indent=2));print(rows)
