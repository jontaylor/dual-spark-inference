"""Experimental draft-only transforms. Target sampling remains untouched."""
import torch
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

def transform_draft_logits(logits, idx_mapping, temperature, target_top_k, target_top_p,
                           mode='top_k', temperature_multiplier=1.0):
    # Return pre-request-temperature logits: the unmodified gumbel sampler
    # caches exactly these scores and the verifier applies request temperature.
    # Keeping the same dtype avoids a sampling/cache rounding discrepancy.
    if temperature_multiplier != 1.0:
        logits = logits / temperature_multiplier
    if mode == 'none':
        return logits
    if mode == 'fixed_top20':
        # A draft distribution may use a fixed shortlist independently of the
        # target's filter. Keeping ties matches threshold-based top-k semantics.
        values = torch.topk(logits, min(20, logits.shape[-1]), dim=-1).values
        return logits.masked_fill(logits < values[:, -1:], -float('inf'))
    if mode == 'fixed_top20_top_p':
        # A fixed 20-token proposal shortlist; nucleus filtering needs only
        # these scores. Exact proposal probabilities are still cached below.
        values, indices = torch.topk(logits, min(20, logits.shape[-1]), dim=-1)
        slots = idx_mapping.clamp_min(0).long()
        temp = temperature[slots].float()
        temp = torch.where(temp > 0, temp, torch.ones_like(temp))
        probs = (values.float() / temp.unsqueeze(1)).softmax(-1)
        p = target_top_p[slots].clamp(0.000001, 1.0).unsqueeze(1)
        keep = probs.cumsum(-1) - probs < p
        keep[:, 0] = True
        mask = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, indices, keep)
        return logits.masked_fill(~mask, -float('inf'))
    if mode not in ('top_k', 'top_k_top_p'):
        raise ValueError(mode)
    # Padding uses index -1 in graph buckets. Keep it address-safe; the sampler
    # already excludes invalid request rows from cache writes.
    slots=idx_mapping.clamp_min(0).long()
    k=target_top_k[slots].clamp(1,logits.shape[-1])
    if mode == 'top_k':
        # General top-k using the existing validated runtime implementation.
        filtered=apply_top_k_top_p(logits.float(),k,None)
        return logits.masked_fill(torch.isneginf(filtered),-float('inf'))
    # Nucleus membership depends on temperature. Determine the mask at that
    # temperature, then apply it to PRE-temperature scores rather than storing
    # scores that the verifier would divide a second time.
    temp=temperature[slots].float()
    temp=torch.where(temp>0,temp,torch.ones_like(temp))
    scaled=logits.float()/temp.unsqueeze(1)
    p=target_top_p[slots].clamp(0.000001,1.0)
    filtered=apply_top_k_top_p(scaled,k,p)
    return logits.masked_fill(torch.isneginf(filtered),-float('inf'))
