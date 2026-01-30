from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

import argparse

def format_boolq(ex):
    passage = ex["passage"].strip()
    question = ex["question"].strip()

    return (
        f"Passage: {passage}\n"
        f"Question: {question}\n"
        f"Answer:"
    )

parser = argparse.ArgumentParser()
parser.add_argument("--bits", default=4, type=int)
parser.add_argument("--group_size", default=128, type=int)
args = parser.parse_args()

model_id = "meta-llama/Llama-2-7b-hf"
quant_path = f"outputs/Llama-2-7b-hf-gptqmodel-boolq-{args.bits}bit-{args.group_size}g"

ds = load_dataset("boolq", split="train").select(range(128))
ds = ds.map(lambda ex: {"text": format_boolq(ex)}, remove_columns=ds.column_names)
calibration_dataset = ds["text"]  # list[str]

quant_config = QuantizeConfig(bits=args.bits, group_size=args.group_size)

model = GPTQModel.load(model_id, quant_config)

# increase `batch_size` to match GPU/VRAM specs to speed up quantization
model.quantize(calibration_dataset, batch_size=1)

model.save(quant_path)