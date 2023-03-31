import glob
import datasets
from tqdm import tqdm
import numpy as np

bleu = datasets.load_metric('sacrebleu')

def make_pred(actual_file, pred_file, ret_metrics):
    # read in the actual and predicted ingredient files
    with open(actual_file, 'r') as f:
        actual_instruction = ""
        actual_doc = f.readlines()
        for line in actual_doc:
            actual_instruction = actual_instruction + line.strip() + " "

    with open(pred_file, 'r') as f:
        predicted_instruction = ""
        predicted_doc = f.readlines()
        for line in predicted_doc:
            predicted_instruction = predicted_instruction + line.strip() + " "
        if predicted_instruction == '-1 ':
            return

    # print(actual_instruction)
    # print()
    # print(predicted_instruction)
    score = bleu.compute(predictions=[[predicted_instruction]], references = [[[actual_instruction]]])
    ret_metrics['bleu'].append(score['score'])


def main():
    GT  = sorted(glob.glob('TEST_DATASET/GT/instructions/*'))
    PRED  = sorted(glob.glob('TEST_DATASET/Predicted/instructions/*'))
    ret_metrics = {'bleu': []}
    
    for actual_file, pred_file in tqdm(zip(GT, PRED)):
        make_pred(actual_file, pred_file, ret_metrics)
        
    for k, v in ret_metrics.items():
        ret_metrics[k] = np.mean(v)
    
    print(ret_metrics)
    
if __name__ == "__main__":
    main()