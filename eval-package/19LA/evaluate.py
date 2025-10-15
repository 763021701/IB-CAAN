import os
import sys
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf, test_individual_attacks

submit_file = sys.argv[1]
op = sys.argv[2]

if __name__ == "__main__":
    if not os.path.isfile(submit_file):
        print("%s doesn't exist" % (submit_file))
        exit(1)

    with open('ASVspoof2019.LA.cm.eval.trl.txt', 'r', encoding='utf-8') as protocol_file:
        protocol_lines = protocol_file.readlines()

    with open(submit_file, 'r', encoding='utf-8') as original_scores_file:
        original_score_lines = original_scores_file.readlines()

    with open('eval_cm_scores_19LA.txt', 'w', encoding='utf-8') as new_socres_file:
        for p_line, s_line in zip(protocol_lines, original_score_lines):
            column_p = p_line.strip().split()
            column_s = s_line.strip().split()
            column_s.insert(1, column_p[3])
            column_s.insert(2, column_p[4])
            updated_line = ' '.join(column_s)
            new_socres_file.write(updated_line + '\n')

    if op == 'i':
        test_individual_attacks('eval_cm_scores_19LA.txt', '.')
    else:
        compute_eer_and_tdcf('eval_cm_scores_19LA.txt', '.')
