import sys
import torch
from transformers import AutoTokenizer
from gptqmodel import GPTQModel

import lm_eval
from lm_eval import utils as lm_eval_utils
from lm_eval.tasks import TaskManager
from lm_eval.models.huggingface import HFLM

model_id = sys.argv[1]
tasks = ["piqa", "winogrande"]

# 1) load tokenizer / model
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = GPTQModel.load(model_id)

# (옵션) device / dtype
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model.to(device)
except Exception:
    pass  # GPTQModel이 내부적으로 device 처리하면 무시

# 2) lm-eval wrapper (hflm 만들기)
hflm = HFLM(
    pretrained=model,          # HF PreTrainedModel 호환이면 OK
    tokenizer=tokenizer,
    device=device,
)

# 3) TaskManager (lm_eval.tasks 말고 직접 import한 TaskManager 사용)
task_manager = TaskManager(
    include_defaults=True,
)

task_names = lm_eval_utils.pattern_match(tasks, task_manager.all_tasks)

results = {}
out = lm_eval.simple_evaluate(
    model=hflm,
    tasks=task_names,
    batch_size=64,
    task_manager=task_manager,
)

for task_name in task_names:
    result = out["results"][task_name]
    acc = round(result.get("acc_norm,none", result["acc,none"]) * 100, 2)
    results[task_name] = acc
    print(f"{task_name}: {acc}%")

metric_vals = dict(results)
metric_vals["acc_avg"] = round(sum(metric_vals.values()) / len(metric_vals.values()), 2)
print(metric_vals)