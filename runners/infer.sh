# 5.86
python examples/benchmark/perplexity.py \
        --model outputs/Llama-2-7b-hf-gptqmodel-3bit \
        --trust_remote_code \
        --is_quantized

python examples/benchmark/perplexity.py \
        --model outputs/Llama-2-7b-hf-gptqmodel-4bit \
        --trust_remote_code \
        --is_quantized