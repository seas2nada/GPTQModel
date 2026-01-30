from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

def format_boolq(ex):
    passage = ex["passage"].strip()
    question = ex["question"].strip()

    return (
        f"Passage: {passage}\n"
        f"Question: {question}\n"
        f"Answer:"
    )

model_id = "meta-llama/Llama-2-7b-hf"
quant_path = "outputs/Llama-2-7b-hf-gptqmodel-boolq-4bit"

ds = load_dataset("boolq", split="train").select(range(128))
ds = ds.map(lambda ex: {"text": format_boolq(ex)}, remove_columns=ds.column_names)
calibration_dataset = ds["text"]  # list[str]

quant_config = QuantizeConfig(bits=4, group_size=128)

model = GPTQModel.load(model_id, quant_config)

# increase `batch_size` to match GPU/VRAM specs to speed up quantization
model.quantize(calibration_dataset, batch_size=1)

model.save(quant_path)