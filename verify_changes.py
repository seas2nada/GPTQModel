
import os
import shutil
from unittest.mock import MagicMock
import sys

# Mocking necessary modules
sys.modules['transformers'] = MagicMock()
sys.modules['gptqmodel'] = MagicMock()
# from gptqmodel.utils.perplexity import Perplexity

def test_saving_logic():
    print("Testing saving logic...")
    test_dir = "test_model_dir"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    # Simulated data (simulating cumulative PPL trace)
    ppl_results = [10.0, 15.0, 12.0]
    
    # Logic to test: Last element is the corpus PPL
    avg_ppl = ppl_results[-1]
    print(f"Calculated Perplexity (Last Element): {avg_ppl}")
    
    args = MagicMock()
    args.model = test_dir
    
    if os.path.isdir(args.model):
        with open(os.path.join(args.model, "perplexity.txt"), "w") as f:
            f.write(str(avg_ppl))
            
    # Verify
    expected_file = os.path.join(test_dir, "perplexity.txt")
    if os.path.exists(expected_file):
        with open(expected_file, "r") as f:
            content = f.read()
            print(f"File content: {content}")
            if content == str(avg_ppl):
                print("SUCCESS: File created with correct content.")
            else:
                print("FAILURE: Incorrect content.")
    else:
        print("FAILURE: File not created.")

    # Cleanup
    shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_saving_logic()
