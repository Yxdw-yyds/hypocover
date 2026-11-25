"""Four paired training-only interventions. Run with -m c4.order_probe [--live]."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

from c4.run_layered import DeepSeek, load_questions
from run_3d_benchmark import Structure3D


def parse(bench, text):
    structure = bench.parse_llm_response(text)
    return structure.normalize() if structure is not None else None


def prepare():
    histories = json.loads(Path(__file__).with_name("order_probe_histories.json").read_text())
    cases = []
    questions = sorted(load_questions("train"), key=lambda q: q["id"])
    for q in questions:
        if q["id"] not in histories:
            continue
        structures, counts = {}, Counter()
        bench = q["bench"]
        for i, layers in enumerate(histories[q["id"]], 1):
            structure = Structure3D(layers) if layers is not None else None
            if structure is None:
                continue
            key = structure.get_hash()
            structures.setdefault(key, structure)
            counts[key] += 1
            order = list(structures)
            top = max(order, key=counts.get)
            if i < q["budget"] // 2 or len(order) < 3 or counts[top] < 2 or top == order[-1]:
                continue
            item = next(x for x in bench.all_observation_sets if x["observation_id"] == q["id"])
            gt = set()
            for answer in item["ground_truth_structures"]:
                layers = answer["layers"]
                if isinstance(layers[0], str):
                    n = int(len(layers[0]) ** 0.5)
                    layers = [[[int(c) for c in layer[j:j+n]] for j in range(0,len(layer),n)] for layer in layers]
                gt.add(Structure3D(layers).get_hash())
            if not gt.difference(order):
                continue
            orders = {"original": order, "reordered": [x for x in order if x != top] + [top]}
            prompts = {arm: bench.create_prompt(q["obs"], [structures[x] for x in keys]) for arm, keys in orders.items()}
            assert prompts["original"] != prompts["reordered"]
            cases.append(dict(id=q["id"], k=q["k"], slot=i+1, prompts=prompts, bench=bench, remaining=gt.difference(order)))
            break
    assert len(cases) == 4
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    live = parser.parse_args().live
    cases = prepare()
    print([(c["id"], c["slot"]) for c in cases], flush=True)
    if not live:
        return
    api = DeepSeek(1024, "deepseek-flash", "disabled", 0, 0, "https://api.deepseek.com")
    output = Path(__file__).with_name("order_probe_results.jsonl")
    first = ["original", "reordered"] * 2
    random.Random(1125).shuffle(first)
    with output.open("x", encoding="utf-8") as f:
        for c, arm1 in zip(cases, first):
            for arm in (arm1, "reordered" if arm1 == "original" else "original"):
                rec = api.query(c["prompts"][arm], {})
                st = parse(c["bench"], rec["text"])
                win = st is not None and st.get_hash() in c["remaining"]
                row = dict(question=c["id"], slot=c["slot"], arm=arm, new=win,
                           finish_reason=rec["finish_reason"], prompt_tokens=rec["prompt_tokens"],
                           completion_tokens=rec["completion_tokens"])
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(c["id"], arm, win, rec["finish_reason"], flush=True)


if __name__ == "__main__":
    main()
