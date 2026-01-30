# WikiText2
# python examples/benchmark/perplexity.py \
#         --model outputs/Llama-2-7b-hf-gptqmodel-3bit \
#         --trust_remote_code \
#         --is_quantized

# python examples/benchmark/perplexity.py \
#         --model outputs/Llama-2-7b-hf-gptqmodel-4bit \
#         --trust_remote_code \
#         --is_quantized

# BoolQ
# python examples/benchmark/perplexity.py \
#         --model outputs/Llama-2-7b-hf-gptqmodel-boolq-4bit \
#         --trust_remote_code \
#         --is_quantized

# python examples/benchmark/perplexity.py \
#         --model outputs/Llama-2-7b-hf-gptqmodel-boolq-3bit \
#         --trust_remote_code \
#         --is_quantized

python examples/benchmark/perplexity.py \
        --model outputs/Llama-2-7b-hf-gptqmodel-piqa-4bit \
        --trust_remote_code \
        --is_quantized

python examples/benchmark/perplexity.py \
        --model outputs/Llama-2-7b-hf-gptqmodel-piqa-3bit \
        --trust_remote_code \
        --is_quantized

python examples/benchmark/perplexity.py \
        --model outputs/Llama-2-7b-hf-gptqmodel-winogrande-4bit \
        --trust_remote_code \
        --is_quantized

python examples/benchmark/perplexity.py \
        --model outputs/Llama-2-7b-hf-gptqmodel-winogrande-3bit \
        --trust_remote_code \
        --is_quantized
