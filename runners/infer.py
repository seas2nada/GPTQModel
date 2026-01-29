# import sys
# from gptqmodel import GPTQModel
# from gptqmodel.utils.eval import EVAL

# quant_path = sys.argv[1]

# # Use `lm-eval` as framework to evaluate the model
# lm_eval_data = GPTQModel.eval(quant_path, 
#                     framework=EVAL.LM_EVAL, 
#                     tasks=[EVAL.LM_EVAL.ARC_CHALLENGE])


# # Use `evalplus` as framework to evaluate the model
# evalplus_data = GPTQModel.eval(quant_path, 
#                     framework=EVAL.EVALPLUS, 
#                     tasks=[EVAL.EVALPLUS.HUMAN])
