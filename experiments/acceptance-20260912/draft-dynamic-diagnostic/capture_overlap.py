"""Bounded diagnostic snapshots, outside graph capture; disabled by sentinel.

Captures only logits/slot metadata, no prompt text. Timing windows containing
captures are diagnostic and must not be used to claim throughput improvements.
"""
import json,pathlib,time
import torch

class OverlapCapture:
    def __init__(self):
        from vllm.distributed import get_tensor_model_parallel_rank
        self.enabled=get_tensor_model_parallel_rank()==0
        self.root=pathlib.Path('/tmp/mtp-overlap-capture')
        self.count=0;self.last=0.0
        if self.enabled:self.root.mkdir(exist_ok=True)

    def capture(self,target,draft,cu,idx,idx_np,positions,temperature,num_sampled,top_k=None,top_p=None,draft_multipliers=None):
        if not self.enabled or draft is None or self.count>=192:return
        now=time.time()
        if now-self.last<2 or (self.root/'STOP').exists():return
        if torch.cuda.is_current_stream_capturing():return
        # cu has at most one small entry per request; transfer only on snapshot.
        bounds=cu.cpu().tolist()
        eligible=[i for i in range(len(idx_np)) if bounds[i+1]-bounds[i]>=4 and int(idx_np[i])>=0]
        if not eligible:return
        row=eligible[self.count%len(eligible)];slot=int(idx_np[row]);lo=bounds[row]
        self.last=now
        if draft_multipliers is not None and float(draft_multipliers[slot].item()) != 1.0:
            return
        data={'target_processed':target[lo:lo+3].detach().cpu().clone(),
              'draft_cached':draft[slot,:3].detach().cpu().clone(),
              'temperature':float(temperature[slot].item()),
              'positions':positions[lo:lo+3].cpu().clone(),
              'num_sampled':int(num_sampled[row].item()),
              'time':now,'slot':slot,'snapshot':self.count,
              'top_k':int(top_k[slot].item()) if top_k is not None else None,
              'top_p':float(top_p[slot].item()) if top_p is not None else None}
        torch.save(data,self.root/f'{self.count:04d}.pt')
        self.count+=1
