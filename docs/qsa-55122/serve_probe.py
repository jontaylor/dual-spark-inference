"""Matched serving probes; background agent load remains running in both arms."""

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

import aiohttp
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
DEPLOY = Path("/home/jon/dual-spark-inference-kv-paging")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    args = parser.parse_args()
    cfg = json.loads((DEPLOY / "deploy_config.json").read_text())
    key = Path(cfg["api_key_file"]).expanduser().read_text().strip()
    snapshot = (
        Path(cfg["paths"]["hf_cache"]).expanduser()
        / ("models--" + cfg["model_id"].replace("/", "--"))
        / "snapshots"
        / cfg["revision"]
    )
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    source = (ROOT / "upstream-head-persistent_topk.cuh").read_text()
    corpus = tokenizer.encode(source * 12, add_special_tokens=False).ids
    output = ROOT / (args.arm + "-serving.jsonl")
    run = str(time.time_ns())
    headers = {"Authorization": "Bearer " + key}
    async with aiohttp.ClientSession(
        headers=headers, timeout=aiohttp.ClientTimeout(total=300)
    ) as session:

        async def request(case, group, target, suffix):
            marker = "QSA-" + hashlib.sha256(group.encode()).hexdigest()[:10]
            # A unique leading session marker prevents prior-arm cache reuse.
            lead = f"Session {run}-{group}. Read the source excerpt and remember its verification marker.\n"
            end = (
                "\nReturn the VERIFICATION_MARKER value, then explain in two sentences "
                "why overflow in top-k candidate buffers is a correctness problem. "
                f"Request suffix: {suffix}."
            )
            n = target - 200
            for _ in range(4):
                content = (
                    lead
                    + tokenizer.decode(corpus[: n // 2])
                    + "\nVERIFICATION_MARKER="
                    + marker
                    + "\n"
                    + tokenizer.decode(corpus[n // 2 : n])
                    + end
                )
                messages = [{"role": "user", "content": content}]
                async with session.post(
                    "http://127.0.0.1:30001/tokenize",
                    json={
                        "model": cfg["served_names"][0],
                        "messages": messages,
                        "add_generation_prompt": True,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                ) as response:
                    response.raise_for_status()
                    tok = await response.json()
                count = tok.get("count", len(tok.get("tokens", [])))
                if abs(count - target) < 4:
                    break
                n += target - count
            payload = {
                "model": cfg["served_names"][0],
                "messages": messages,
                "temperature": 0,
                "top_p": 1,
                "seed": 55122,
                "max_tokens": 96,
                "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False},
            }
            start = time.monotonic()
            first = None
            text = ""
            usage = None
            metrics = None
            finish = None
            async with session.post(
                "http://127.0.0.1:30001/v1/chat/completions", json=payload
            ) as response:
                response.raise_for_status()
                async for raw in response.content:
                    line = raw.decode().strip()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    data = json.loads(line[6:])
                    usage = data.get("usage") or usage
                    metrics = data.get("metrics") or metrics
                    for choice in data.get("choices", []):
                        delta = choice.get("delta", {})
                        piece = (
                            delta.get("content") or delta.get("reasoning_content") or ""
                        )
                        if piece and first is None:
                            first = time.monotonic()
                        text += piece
                        finish = choice.get("finish_reason") or finish
            stop = time.monotonic()
            result = dict(
                arm=args.arm,
                case=case,
                group=group,
                target=target,
                time=time.time(),
                ttft_s=None if first is None else first - start,
                total_s=stop - start,
                usage=usage,
                metrics=metrics,
                text=text,
                marker=marker,
                marker_present=marker in text,
                finish=finish,
                prompt_count=count,
                decode_tps=(usage["completion_tokens"] - 1) / (stop - first)
                if usage and first and stop > first
                else None,
            )
            with output.open("a") as f:
                f.write(json.dumps(result) + "\n")
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in (
                            "arm",
                            "case",
                            "target",
                            "ttft_s",
                            "total_s",
                            "marker_present",
                        )
                    }
                ),
                flush=True,
            )
            return result

        # Each suffix has the same short token budget to keep the shared prefix stable.
        for i in range(3):
            await request("cold20k", f"prefix-{i}", 20000, "A")
            await request("reuse20k", f"prefix-{i}", 20000, "B")
        await request("cold32k", "long", 32768, "A")
        await asyncio.gather(
            *(request("c4-8k", f"c4-{i}", 8192, "A") for i in range(4))
        )


if __name__ == "__main__":
    asyncio.run(main())
