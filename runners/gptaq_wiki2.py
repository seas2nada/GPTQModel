from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.quantization import GPTAQConfig

model_id = "meta-llama/Llama-2-7b-hf"
quant_path = "outputs/Llama-2-7b-hf-gptaqmodel-4bit"

calibration_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train").select(range(128))["text"]

quant_config = QuantizeConfig(bits=4, group_size=128, gptaq=GPTAQConfig(alpha=0.25, device="auto"))

model = GPTQModel.load(model_id, quant_config)

# increase `batch_size` to match GPU/VRAM specs to speed up quantization
model.quantize(calibration_dataset, batch_size=1)

model.save(quant_path)