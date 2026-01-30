from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

def format_piqa(ex):
    goal = ex["goal"].strip()
    sol1 = ex["sol1"].strip()
    sol2 = ex["sol2"].strip()
    # PIQA는 2-choice라 A/B로 두는 게 깔끔
    return (
        f"Goal: {goal}\n"
        f"Choices:\n"
        f"A. {sol1}\n"
        f"B. {sol2}\n"
        f"Answer:"
    )

model_id = "meta-llama/Llama-2-7b-hf"
quant_path = "outputs/Llama-2-7b-hf-gptqmodel-piqa-3bit"

ds = load_dataset("baber/piqa", split="train", trust_remote_code=True).select(range(128))
ds = ds.map(lambda ex: {"text": format_piqa(ex)}, remove_columns=ds.column_names)
calibration_dataset = ds["text"]  # list[str]

quant_config = QuantizeConfig(bits=3, group_size=128)

model = GPTQModel.load(model_id, quant_config)

# increase `batch_size` to match GPU/VRAM specs to speed up quantization
model.quantize(calibration_dataset, batch_size=1)

model.save(quant_path)