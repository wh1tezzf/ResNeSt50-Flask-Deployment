import argparse
import sys
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

#parameters
debug = False

###########
#functions#
###########

def convert_label(label):
    bird_code = {
    'aldfly': 0, 'amewoo':1, 'bktspa':2, 'brncre':3, 'bulori':4, 'canwre':5,
    'clanut':6, 'comyel':7, 'easkin':8, 'foxspa':9,'greegr':10, 'hergul':11,
    'lesnig':12, 'magwar':13, 'normoc':14, 'palwar':15, 'plsvir':16, 'rethaw':17,
    'rufgro':18, 'stejay':19, 'norcar':20, 'haiwoo':21, 'amered':22,'easblu':23, 'treswa':24
    }
    return bird_code[label];

######
#main#
######

def main(args):

    try:
        groundtruth = [item.strip() for item in open(args.groundtruth, "r").readlines()]
        predictions = [item.strip() for item in open(args.predictions, "r").readlines()]

        if (len(groundtruth)!=len(predictions)):
            raise Exception("groundtruth file must have the same number of lines as the predictions file")

        y_true, y_pred = [], []

        for vi, v in enumerate(groundtruth):
            y_pred.append(convert_label(predictions[vi]))
            y_true.append(convert_label(v))

        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average="macro")
        accuracy = accuracy_score(y_true, y_pred)
    except Exception as error:
        print("Error:", error)
        raise SystemExit


    print("Performance on the rumour class:")
    print("Precision =", p)
    print("Recall    =", r)
    print("F1        =", f)
    print("Accuracy  =", accuracy)
        

if __name__ == "__main__":

    #parser arguments
    desc = "Computes precision, recall and F-score of the rumour class"
    parser = argparse.ArgumentParser(description=desc)

    #arguments
    parser.add_argument("--predictions", required=True, help="text file containing system predictions (one line per label)")
    parser.add_argument("--groundtruth", required=True, help="text file containing ground truth labels (one line per label)")
    args = parser.parse_args()

    main(args)
