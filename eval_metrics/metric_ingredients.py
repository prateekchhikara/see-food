import numpy as np
import pickle
import glob

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
    def add_word(self, word, idx=None):
        if idx is None:
            if not word in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1
            return self.idx
        else:
            if not word in self.word2idx:
                self.word2idx[word] = idx
                if idx in self.idx2word.keys():
                    self.idx2word[idx].append(word)
                else:
                    self.idx2word[idx] = [word]
                    
                return idx
            
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<pad>']
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.idx2word)
    
def softIoU(out, target, e=1e-6, sum_axis=1):
    num = (out*target).sum()
    den = (out+target-out*target).sum() + e
    iou = num / den
    return iou
    
def compute_metrics(ret_metrics, error_types, metric_names, eps=1e-10, weights=None):
    if 'accuracy' in metric_names:
        ret_metrics['accuracy'].append(np.mean((error_types['tp_i'] + error_types['tn_i']) / (error_types['tp_i'] + error_types['fp_i'] + error_types['fn_i'] + error_types['tn_i'])))
    if 'jaccard' in metric_names:
        ret_metrics['jaccard'].append(error_types['tp_all'] / (error_types['tp_all'] + error_types['fp_all'] + error_types['fn_all'] + eps))
    if 'dice' in metric_names:
        ret_metrics['dice'].append(2*error_types['tp_all'] / (2*(error_types['tp_all'] + error_types['fp_all'] + error_types['fn_all']) + eps))
    if 'f1' in metric_names:
        pre = error_types['tp_i'] / (error_types['tp_i'] + error_types['fp_i'] + eps)
        rec = error_types['tp_i'] / (error_types['tp_i'] + error_types['fn_i'] + eps)
        f1_perclass = 2*(pre * rec) / (pre + rec + eps)
        if 'f1_ingredients' not in ret_metrics.keys():
            ret_metrics['f1_ingredients'] = [np.average(f1_perclass, weights=weights)]
        else:
            ret_metrics['f1_ingredients'].append(np.average(f1_perclass, weights=weights))

        pre = error_types['tp_all'] / (error_types['tp_all'] + error_types['fp_all'] + eps)
        rec = error_types['tp_all'] / (error_types['tp_all'] + error_types['fn_all'] + eps)
        f1 = 2*(pre * rec) / (pre + rec + eps)
        ret_metrics['f1'].append(f1)
        
        
def update_error_types(error_types, y_pred, y_true):
    error_types['tp_i'] += (y_pred * y_true).sum(0)
    error_types['fp_i'] += (y_pred * (1-y_true)).sum(0)
    error_types['fn_i'] += ((1-y_pred) * y_true).sum(0)
    error_types['tn_i'] += ((1-y_pred) * (1-y_true)).sum(0)
    error_types['tp_all'] += (y_pred * y_true).sum()
    error_types['fp_all'] += (y_pred * (1-y_true)).sum()
    error_types['fn_all'] += ((1-y_pred) * y_true).sum()
    
    
def make_pred(vocab, actual_file, pred_file, ret_metrics):
    # read in the actual and predicted ingredient files
    with open(actual_file, 'r') as f:
        actual_ingredients = f.readlines()
        actual_ingredients = [ingr.strip() for ingr in actual_ingredients]
    with open(pred_file, 'r') as f:
        predicted_ingredients = f.readlines()
        predicted_ingredients = [ingr.strip() for ingr in predicted_ingredients]
        if predicted_ingredients[0] == '-1':
            # print(pred_file)
            return
    
        
    y_true = np.zeros(len(vocab.idx2word))
    for ingr in actual_ingredients:
        index = vocab.word2idx.get(ingr, -1)
        if index > 0:
            y_true[index] = 1
            
    # print(f'y_true hits: {(y_true == 1).sum()}')
    
    y_pred = np.zeros(len(vocab.idx2word))
    for ingr in predicted_ingredients:
        index = vocab.word2idx.get(ingr, -1)
        # print(index)
        if index > 0:
            y_pred[index] = 1

    # print(f'y_pred hits: {(y_pred == 1).sum()}')

    error_types = {'tp_i': 0, 'fp_i': 0, 'fn_i': 0, 'tn_i': 0, 'tp_all': 0, 'fp_all': 0, 'fn_all': 0}
    update_error_types(error_types, y_pred, y_true)

    compute_metrics(ret_metrics, error_types, ['accuracy', 'f1', 'jaccard', 'f1_ingredients', 'dice'],
                            eps=1e-10,
                            weights=None)

    # iou_item = np.mean(softIoU(y_pred, y_true)).item()


def main():
    vocab = pickle.load(open('/data/prateek/github/see-food/garbage/recipe1m_vocab_ingrs.pkl', 'rb'))
    # actual_file = 'actual_ingredients.txt'
    # pred_file = 'predicted_ingredients.txt'
    # ret_metrics = make_pred(vocab, actual_file, pred_file)
    # print(ret_metrics)

    GT  = sorted(glob.glob('../TEST_DATASET/GT/ingredients/*'))
    PRED  = sorted(glob.glob('../TEST_DATASET/PRED-baseline/ingredients/*'))
    ret_metrics = {'accuracy': [], 'f1': [], 'jaccard': [], 'f1_ingredients': [], 'dice' : []}

    for actual_file, pred_file in zip(GT, PRED):
        make_pred(vocab, actual_file, pred_file, ret_metrics)
        
    # print(ret_metrics)

    
    for k, v in ret_metrics.items():
        ret_metrics[k] = round(100*np.mean(v), 3)
    
    print(ret_metrics)

if __name__ == "__main__":
    main()