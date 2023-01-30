import collections
import math



def get_dist(res):
    unigrams = []
    bigrams = []
    avg_len = 0.
    ma_dist1, ma_dist2 = 0., 0.
    for r in res:
        ugs = r.split()
        bgs = []
        i = 0
        while i < len(ugs) - 1:
            bgs.append(ugs[i] + ugs[i + 1])
            i += 1
        unigrams += ugs
        bigrams += bgs
        ma_dist1 += len(set(ugs)) / (float)(len(ugs) + 1e-16)
        ma_dist2 += len(set(bgs)) / (float)(len(bgs) + 1e-16)
        avg_len += len(ugs)
    n = len(res)
    ma_dist1 /= n
    ma_dist2 /= n
    mi_dist1 = len(set(unigrams)) / (float)(len(unigrams))
    mi_dist2 = len(set(bigrams)) / (float)(len(bigrams))
    avg_len /= n
    return ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len

def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.

    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.

    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts

def compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False):
    """Computes BLEU score of translated segments against one or more references.

    Args:
      reference_corpus: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      smooth: Whether or not to apply Lin et al. 2004 smoothing.

    Returns:
      3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
      precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                         translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / (ratio + 1e-16))

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)

def get_bleu(cands, refs):
    assert len(cands) == len(refs)

    ma_bleu = 0.
    ma_bleu1 = 0.
    ma_bleu2 = 0.
    ma_bleu3 = 0.
    ma_bleu4 = 0.
    ref_lst = []
    hyp_lst = []
    for q, r in enumerate(cands):
        references = refs[q].split()
        hypothesis = r.split()
        ref_lst.append(references)
        hyp_lst.append(hypothesis)
        bleu, precisions, _, _, _, _ = compute_bleu([references], [hypothesis], smooth=False)
        ma_bleu += bleu
        ma_bleu1 += precisions[0]
        ma_bleu2 += precisions[1]
        ma_bleu3 += precisions[2]
        ma_bleu4 += precisions[3]
    n = len(cands)
    ma_bleu /= n
    ma_bleu1 /= n
    ma_bleu2 /= n
    ma_bleu3 /= n
    ma_bleu4 /= n

    mi_bleu, precisions, _, _, _, _ = compute_bleu(ref_lst, hyp_lst, smooth=False)
    mi_bleu1, mi_bleu2, mi_bleu3, mi_bleu4 = precisions[0], precisions[1], precisions[2], precisions[3]
    return ma_bleu, ma_bleu1, ma_bleu2, ma_bleu3, ma_bleu4, \
           mi_bleu, mi_bleu1, mi_bleu2, mi_bleu3, mi_bleu4

def get_dict(tokens, ngram, gdict=None):
    """
    get_dict
    """
    token_dict = {}
    if gdict is not None:
        token_dict = gdict
    tlen = len(tokens)
    for i in range(0, tlen - ngram + 1):
        ngram_token = "".join(tokens[i:(i + ngram)])
        if token_dict.get(ngram_token) is not None:
            token_dict[ngram_token] += 1
        else:
            token_dict[ngram_token] = 1
    return token_dict


def count(pred_tokens, gold_tokens, ngram, result):
    """
    count
    """
    cover_count, total_count = result
    pred_dict = get_dict(pred_tokens, ngram)
    gold_dict = get_dict(gold_tokens, ngram)
    cur_cover_count = 0
    cur_total_count = 0
    for token, freq in pred_dict.items():
        if gold_dict.get(token) is not None:
            gold_freq = gold_dict[token]
            cur_cover_count += min(freq, gold_freq)
        cur_total_count += freq
    result[0] += cur_cover_count
    result[1] += cur_total_count


def calc_bp(pair_list):
    """
    calc_bp
    """
    c_count = 0.0
    r_count = 0.0
    for pair in pair_list:
        pred_tokens, gold_tokens = pair
        c_count += len(pred_tokens)
        r_count += len(gold_tokens)
    bp = 1
    if c_count < r_count:
        bp = math.exp(1 - r_count / c_count)
    return bp

def calc_cover_rate(pair_list, ngram):
    """
    calc_cover_rate
    """
    result = [0.0, 0.0] # [cover_count, total_count]
    for pair in pair_list:
        pred_tokens, gold_tokens = pair
        count(pred_tokens, gold_tokens, ngram, result)
    cover_rate = (result[0]  + 0.0001) / (result[1] + 0.0001)
    return cover_rate


def calc_bleu(pair_list):
    """
    calc_bleu
    """
    bp = calc_bp(pair_list)
    cover_rate1 = calc_cover_rate(pair_list, 1)
    cover_rate2 = calc_cover_rate(pair_list, 2)
    cover_rate3 = calc_cover_rate(pair_list, 3)
    cover_rate4 = calc_cover_rate(pair_list, 4)
    bleu1 = 0
    bleu2 = 0
    bleu3 = 0
    bleu4 = 0
    if cover_rate1 > 0:
        bleu1 = bp * math.exp(math.log(cover_rate1))
    if cover_rate2 > 0:
        bleu2 = bp * math.exp((math.log(cover_rate1) + math.log(cover_rate2)) / 2)
    if cover_rate3 > 0:
        bleu3 = bp * math.exp((math.log(cover_rate1) + math.log(cover_rate2) + math.log(cover_rate3)) / 3)
    if cover_rate4 > 0:
        bleu4 = bp * math.exp((math.log(cover_rate1) + math.log(cover_rate2) + math.log(cover_rate3) + math.log(cover_rate4)) / 4)
    avg_bleu = (bleu4 + bleu3 + bleu2 + bleu1) / 4
    return [avg_bleu, bleu1, bleu2, bleu3, bleu4]

