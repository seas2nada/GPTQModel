import sys
from gptqmodel import GPTQModel
from gptqmodel.utils.eval import EVAL

model_id = sys.argv[1]

# Use `lm-eval` as framework to evaluate the model
lm_eval_data = GPTQModel.eval(model_id, 
                    framework=EVAL.LM_EVAL, 
                    tasks=[EVAL.LM_EVAL.BOOLQ],
                    output_path=f"{model_id}/boolq_eval_results.json")