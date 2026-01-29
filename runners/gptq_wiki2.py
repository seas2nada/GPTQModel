from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

model_id = "meta-llama/Llama-2-7b-hf"
quant_path = "Llama-2-7b-hf-gptqmodel-3bit"

calibration_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train").select(range(128))["text"]

quant_config = QuantizeConfig(bits=3, group_size=128)

model = GPTQModel.load(model_id, quant_config)

# increase `batch_size` to match GPU/VRAM specs to speed up quantization
model.quantize(calibration_dataset, batch_size=1)

model.save(quant_path)