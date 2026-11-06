"""Assert revision prompts are byte-identical to the original v4 scripts' prompts.

Pulls the prompt-building code out of the (untouched) original scripts with
``ast`` -- the ``knowledge`` constant, the ``reference_block`` loop, and the
``traffic``/``prompt`` statements inside the per-row loop -- executes it on
every row of the seed-42 subset with the seed-42 reference examples, and
compares against ``prompts_v4``. Also checks that the seed-42 reference
examples equal the ones the original few-shot script selects.
"""

import ast
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import prompts_v4 as P  # noqa: E402

ZERO = "research/llm_classification_zero_shot_v4.py"
FEW = "research/llm_classification_few_shot_v4.py"
SUBSET = "results/revision/setup/llm_subset_v4_seed42.csv"
REFS = "results/revision/setup/fewshot_reference_examples_seeds42-1-2-3-4.json"


def module(path):
    with open(path) as f:
        return ast.parse(f.read())


def assigned(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return node
    raise KeyError(name)


def loop_prompt_code(tree):
    """Statements inside `for i, row in X.iterrows():` up to `prompt = ...`."""
    for node in tree.body:
        if isinstance(node, ast.For) and "iterrows" in ast.unparse(node.iter):
            stmts = []
            for s in node.body:
                stmts.append(s)
                if isinstance(s, ast.Assign) and getattr(s.targets[0], "id", None) == "prompt":
                    return compile(ast.Module(body=stmts, type_ignores=[]), "<orig>", "exec")
    raise RuntimeError("prompt loop not found")


zero_tree, few_tree = module(ZERO), module(FEW)

# constants
assert ast.literal_eval(assigned(few_tree, "selected_features").value) == P.selected_features
assert ast.literal_eval(assigned(zero_tree, "selected_features").value) == P.selected_features
assert ast.literal_eval(assigned(few_tree, "knowledge").value) == P.knowledge
assert ast.literal_eval(assigned(few_tree, "label_names").value) == P.label_names

# reference block built by the original loop from the seed-42 examples
with open(REFS) as f:
    refs = json.load(f)
examples42 = {int(k): v for k, v in refs["42"]["examples"].items()}
ref_loop = next(n for n in few_tree.body if isinstance(n, ast.For)
                and "reference_examples" in ast.unparse(n.iter))
ns = {"reference_examples": examples42, "label_names": P.label_names,
      "reference_block": assigned(few_tree, "reference_block").value.value}
exec(compile(ast.Module(body=[ref_loop], type_ignores=[]), "<orig>", "exec"), ns)
orig_block = ns["reference_block"]
assert orig_block == P.reference_block(examples42)

# per-row prompts
zero_code, few_code = loop_prompt_code(zero_tree), loop_prompt_code(few_tree)
X = pd.read_csv(SUBSET)[P.selected_features]
for i, row in X.iterrows():
    nz = {"row": row, "selected_features": P.selected_features}
    exec(zero_code, nz)
    assert nz["prompt"] == P.zero_shot_prompt(row), f"zero-shot prompt differs at row {i}"
    nf = {"row": row, "selected_features": P.selected_features,
          "knowledge": P.knowledge, "reference_block": orig_block}
    exec(few_code, nf)
    assert nf["prompt"] == P.few_shot_prompt(row, P.reference_block(examples42)), \
        f"few-shot prompt differs at row {i}"

print(f"OK: {len(X)} zero-shot and {len(X)} few-shot prompts byte-identical to the originals.")
print(f"Seed-42 reference rows (train positions): {refs['42']['train_row_position']}")
