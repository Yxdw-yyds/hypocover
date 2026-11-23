"""Score one complete split of candidate answers, without a GPU or model calls."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0,str(root/"upstream/3d"))
from run_3d_benchmark import Benchmark3D

class RecordedAnswers:
    def __init__(self, texts):
        self.texts = iter(texts)
    def get_name(self):
        return "external-candidates"
    def query(self, prompt):
        return next(self.texts)

def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("split",choices=("train","validation","test"))
    args.add_argument("predictions",type=Path,nargs="+",help="One or more JSONL files for the selected split")
    args.add_argument("--output",type=Path,required=True)
    opts = args.parse_args()
    groups = json.loads((root/"data/manifest.json").read_text())["groups"]
    budgets = {q:g["budget"] for g in groups.values() for q in g[opts.split]}
    answers = defaultdict(dict)
    for source in opts.predictions:
        with source.open(encoding="utf-8") as file:
            for row in map(json.loads,file):
                q,slot,text = row["question"],row["slot"],row["scorer_text"]
                if q not in budgets or type(slot) is not int or not 1 <= slot <= budgets[q]:
                    raise ValueError(f"Unexpected question or slot: {q}/{slot}")
                if slot in answers[q] or not isinstance(text,str):
                    raise ValueError(f"Duplicate slot or non-string answer: {q}/{slot}")
                answers[q][slot] = text
    cases = []
    for g in groups.values():
        bench = Benchmark3D(str(root/"data"/g["file"]))
        items = {i["observation_id"]:i for i in bench.all_observation_sets}
        for q in g[opts.split]:
            if len(answers[q]) != budgets[q]:
                raise ValueError(f"{q} has {len(answers[q])}/{budgets[q]} slots; record failed attempts too")
            ordered = [answers[q][i] for i in range(1,budgets[q]+1)]
            case = bench.evaluate_single_observation_set(RecordedAnswers(ordered),items[q],n_queries=len(ordered),
                                                        verbose=False,max_retries=1)
            if case["errors"]:
                raise ValueError(f"Scorer error on {q}: {case['errors']}")
            cases.append(case)
    metrics = ["parse_success_rate","valid_rate","novelty_rate","recovery_rate"]
    summary = {m:mean(c[m] for c in cases) for m in metrics}
    with opts.output.open("x",encoding="utf-8") as file:
        json.dump(dict(split=opts.split,questions=len(cases),summary=summary,cases=cases),file,indent=2)
    print(summary)

if __name__ == "__main__":
    main()
