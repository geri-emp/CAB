from utils.get_metrics import get_dist, calc_bleu
from utils.config import args
import os



def read_file(file_name, dec_pre_type = 'Greedy', dec_ref_type = 'Ref'):
    # print(os.getcwd())
    f = open(f'result/{file_name}/{file_name}.txt', 'r', encoding= 'utf-8')
    # f = open(f'result/prediction/output.txt', 'r', encoding='utf-8')
    # f = open(f'result/EMPDG/results.txt', 'r', encoding='utf-8')
    refs = []
    cands = []
    sentences = []
    dec_pre_str = f"{dec_pre_type}:"
    dec_ref_str = f"{dec_ref_type}:"
    for i, line in enumerate(f.readlines()):
        if line.startswith(dec_pre_str):
            exp = line.strip(dec_pre_str).strip('\n').strip('\n')
            cands.append(exp)
        if line.startswith(dec_ref_str):
            ref = line.strip(dec_ref_str).strip('\n').strip('\n')
            refs.append(ref)
    for cs, rs in zip(cands, refs):
        pred_tokens = cs.strip().split()
        gold_tokens = rs.strip().split()
        sentences.append([pred_tokens, gold_tokens])
    return sentences, cands



if __name__ == '__main__':
    files = ['CAB', 'EMPDG', 'KEMP', 'CEM', 'MIME', 'MOEL', 'Transformer', 'MultiTransformer'] # CAB、KEMP
    # files = ['CAB']
    print_file = args.save_path_+"metrics.txt"
    outputs = open(print_file, 'w', encoding='utf-8')
    ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len = 0, 0, 0, 0, 0
    bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0

    for f in files:
        print(f"Evaluating {f}")
        if f == 'CAB': #Top-k/Top-p Greedy
            sents, cands = read_file(f, dec_pre_type = 'Greedy', dec_ref_type = 'Ref')
        elif f == 'EmpHi': # Greedy
            sents, cands = read_file(f, dec_pre_type = 'Greedy', dec_ref_type = 'Ref')
        elif f == 'EMPDG': # Greedy
            sents, cands = read_file(f, dec_pre_type = 'Greedy', dec_ref_type = 'Ref')
        elif f == 'KEMP': #Pred and TOP
            sents, cands = read_file(f, dec_pre_type = 'Pred', dec_ref_type = 'Ref')
        elif f == 'CEM': #Beam and Greedy AND TOP
            sents, cands = read_file(f, dec_pre_type = 'Greedy', dec_ref_type = 'Ref')
        elif f == 'MIME': #Topk Beam and Greedy
            sents, cands = read_file(f, dec_pre_type='Greedy', dec_ref_type='Ref')
        elif f == 'Transformer' or f == 'MultiTransformer': #Beam and Greedy and TOP
            sents, cands = read_file(f, dec_pre_type='Greedy', dec_ref_type='Ref')
        elif f == 'MOEL':#Beam and Greedy AND TOP
            sents, cands = read_file(f, dec_pre_type = 'Greedy', dec_ref_type = 'Ref')
        ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len = get_dist(cands)
        avg_bleu, bleu1, bleu2, bleu3, bleu4 = calc_bleu(sents)

        print('{}'.format(f), file=outputs)
        print("ma_dist1\tma_dist2\tmi_dist1\tmi_dist2\tavg_len", file=outputs)
        print(
            "{:.4f}\t{:.4f}\t{:.4f}\t{:.4f} \t{:.4f}".format(ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len),
            file=outputs)

        print("avg_bleu\tbleu1\tbleu2\tbleu3\tbleu4", file=outputs)
        print(
            "{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(avg_bleu, bleu1, bleu2, bleu3, bleu4),
            file=outputs)

