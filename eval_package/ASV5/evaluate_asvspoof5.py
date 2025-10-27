#!/usr/bin/env python

import sys, os.path
import numpy as np
import pandas
from eval5.calculate_metrics import calculate_minDCF_EER_CLLR, calculate_aDCF_tdcf_tEER

if len(sys.argv) != 3:
    print("CHECK: invalid input arguments. Please read the instruction below:")
    print(__doc__)
    exit(1)

submit_file = sys.argv[1]
track = sys.argv[2]

def eval_to_score_file(score_file, track):
    eval_dcf, eval_eer, eval_cllr = calculate_minDCF_EER_CLLR(
        cm_scores_file=score_file,
        output_file='./eval_score.txt',
        printout=False
    )
    print("DONE. eval_eer: {:.4f}, eval_dcf: {:.5f} , eval_cllr: {:.5f}".format(
          eval_eer*100, eval_dcf, eval_cllr))
    return eval_dcf, eval_eer, eval_cllr

if __name__ == "__main__":

    if not os.path.isfile(submit_file):
        print("%s doesn't exist" % (submit_file))
        exit(1)

    if track != 'track1' and track != 'track2':
        print("track must be either track1 or track2")
        exit(1)

    _ = eval_to_score_file(submit_file, track)
