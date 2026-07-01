"""Full LongBench v2 (503: short+medium+long), THINKING ENABLED, on the 4-server SGLang fleet (gemma-4-26B-A4B)."""
import json, time, asyncio, re
from pathlib import Path
from datasets import load_dataset
from openai import AsyncOpenAI

PORTS = [30000, 30001, 30002, 30003]
MODEL = "google/gemma-4-26B-A4B-it"
OUT = Path("/tmp/lbv2_full_think_results"); OUT.mkdir(parents=True, exist_ok=True)
PER_SERVER_CONCURRENCY = 16          # lower: long traces are heavy
MAX_TOKENS = 16384                   # flat reasoning+answer budget
clients = [AsyncOpenAI(base_url=f"http://localhost:{p}/v1", api_key="x", timeout=3600) for p in PORTS]

def prompt(it):
    return ("Read the following text and answer the multiple-choice question.\n\n"
        f"{it['context']}\n\nQuestion: {it['question']}\n\n"
        f"A. {it['choice_A']}\nB. {it['choice_B']}\nC. {it['choice_C']}\nD. {it['choice_D']}\n\n"
        "Answer with just the letter (A, B, C, or D).")

def extract_letter(content, reasoning):
    # answer letter is in content (after the reasoning trace); fall back to scanning
    for src in (content or "", reasoning or ""):
        m = re.search(r"\b([ABCD])\b", src)
        if m: return m.group(1)
    return ""

async def run_one(sem, client, it):
    async with sem:
        t = time.perf_counter()
        try:
            r = await client.chat.completions.create(model=MODEL,
                messages=[{"role":"user","content":prompt(it)}],
                max_tokens=MAX_TOKENS, temperature=0.0,
                extra_body={"chat_template_kwargs":{"enable_thinking":True}})
            ch = r.choices[0]; m = ch.message
            pred = extract_letter(m.content, getattr(m, "reasoning_content", None))
            return {"_id":it["_id"],"domain":it["domain"],"length":it["length"],"pred":pred,"gold":it["answer"],
                    "correct":pred.upper()==it["answer"].upper(),"lat":time.perf_counter()-t,
                    "tin":r.usage.prompt_tokens,"tout":r.usage.completion_tokens,
                    "truncated":ch.finish_reason=="length",
                    "reasoning_chars":len(getattr(m,"reasoning_content",None) or "")}
        except Exception as e:
            return {"_id":it["_id"],"domain":it["domain"],"length":it["length"],"pred":None,"gold":it["answer"],
                    "correct":False,"lat":time.perf_counter()-t,"error":str(e)[:200]}

def block(rows):
    n=len(rows); errs=sum(1 for r in rows if r.get("error")); trunc=sum(1 for r in rows if r.get("truncated"))
    c=sum(1 for r in rows if r["correct"])
    return {"n":n,"correct":c,"errors":errs,"truncated":trunc,
            "accuracy":round(c/(n-errs),4) if n-errs>0 else 0}

async def main():
    ds = load_dataset("THUDM/LongBench-v2", split="train")
    items = sorted(list(ds), key=lambda x: len(x["context"]))   # shortest first
    print(f"FULL LongBench-v2 (THINKING, max_tokens={MAX_TOKENS}): {len(items)} tasks, {len(PORTS)} servers", flush=True)
    sems = [asyncio.Semaphore(PER_SERVER_CONCURRENCY) for _ in PORTS]
    t0=time.time()
    tasks=[run_one(sems[i%len(PORTS)], clients[i%len(PORTS)], it) for i,it in enumerate(items)]
    results=await asyncio.gather(*tasks)
    dt=time.time()-t0
    bydom={}
    for r in results: bydom.setdefault(r["domain"],[]).append(r)
    summary={"model":MODEL,"benchmark":"LongBench-v2-full","thinking":True,"max_tokens":MAX_TOKENS,
        "servers":len(PORTS),"overall":block(results),"total_time_s":round(dt,1),
        "throughput_tasks_per_min":round(len(results)/(dt/60),1),
        "input_tokens":sum(r.get("tin",0) for r in results),
        "output_tokens":sum(r.get("tout",0) for r in results),
        "by_length":{L:block([r for r in results if r["length"]==L]) for L in ("short","medium","long")},
        "by_domain":{d:block(rs) for d,rs in bydom.items()}}
    print(json.dumps(summary,indent=2), flush=True)
    json.dump(summary, open(OUT/"summary.json","w"), indent=2)
    with open(OUT/"results.jsonl","w") as f:
        for r in results: f.write(json.dumps(r)+"\n")
    print("saved", OUT, flush=True)

if __name__=="__main__": asyncio.run(main())
