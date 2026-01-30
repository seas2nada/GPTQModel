from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--bits", default=3, type=int)
parser.add_argument("--group_size", default=128, type=int)
args = parser.parse_args()

model_id = "meta-llama/Llama-2-7b-hf"
quant_path = f"Llama-2-7b-hf-gptqmodel-{args.bits}bit-{args.group_size}g"

calibration_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train").select(range(128))["text"]

quant_config = QuantizeConfig(bits=args.bits, group_size=args.group_size)

model = GPTQModel.load(model_id, quant_config)

# increase `batch_size` to match GPU/VRAM specs to speed up quantization
model.quantize(calibration_dataset, batch_size=1)

model.save(quant_path)