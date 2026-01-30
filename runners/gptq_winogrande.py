from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

import argparse

def format_winograde(ex):
    # winogrande: sentence에 '_' 빈칸이 있고 option1/2 중 하나로 채움
    sent = ex["sentence"].strip()
    opt1 = ex["option1"].strip()
    opt2 = ex["option2"].strip()
    return (
        f"Sentence: {sent}\n"
        f"Choices:\n"
        f"A. {opt1}\n"
        f"B. {opt2}\n"
        f"Answer:"
    )

parser = argparse.ArgumentParser()
parser.add_argument("--bits", default=3, type=int)
parser.add_argument("--group_size", default=128, type=int)
args = parser.parse_args()

model_id = "meta-llama/Llama-2-7b-hf"
quant_path = f"outputs/Llama-2-7b-hf-gptqmodel-winogrande-{args.bits}bit-{args.group_size}g"

ds = load_dataset("allenai/winogrande", "winogrande_xl", split="train", trust_remote_code=True).select(range(128))
ds = ds.map(lambda ex: {"text": format_winograde(ex)}, remove_columns=ds.column_names)
calibration_dataset = ds["text"]  # list[str]

quant_config = QuantizeConfig(bits=args.bits, group_size=args.group_size)

model = GPTQModel.load(model_id, quant_config)

# increase `batch_size` to match GPU/VRAM specs to speed up quantization
model.quantize(calibration_dataset, batch_size=1)

model.save(quant_path)