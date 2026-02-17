import numpy as np
from statsmodels.stats.multitest import multipletests


def analyze_performance(rejections, test_y):
    n_outliers = np.sum(test_y == 1)
    n_false_discoveries = np.sum(test_y[rejections] == 0)
    n_true_discoveries = np.sum(test_y[rejections] == 1)
    assert n_true_discoveries + n_false_discoveries == len(rejections)
    power = n_true_discoveries / n_outliers if n_outliers else 1
    fdr = n_false_discoveries / (n_false_discoveries + n_true_discoveries) if (n_false_discoveries + n_true_discoveries) > 0 else 0
    return power, fdr


def compute_conformal_pvalues(null_statistics, test_statistic):
    return (1 + np.sum(null_statistics.reshape((1, -1)) >= test_statistic.reshape((-1, 1)), axis=1)) / (len(null_statistics)+1)


def BH(pvalues, level):
    """
    Benjamini-Hochberg procedure.
    """
    n = len(pvalues)
    pvalues_sort_ind = np.argsort(pvalues)
    pvalues_sort = np.sort(pvalues) #p(1) < p(2) < .... < p(n)

    comp = pvalues_sort <= (level* np.arange(1,n+1)/n)
    #get first location i0 at which p(k) <= level * k / n
    comp = comp[::-1]
    comp_true_ind = np.nonzero(comp)[0]
    i0 = comp_true_ind[0] if comp_true_ind.size > 0 else n
    nb_rej = n - i0
    threshold = pvalues[pvalues_sort_ind[nb_rej - 1]]
    return pvalues_sort_ind[:nb_rej], threshold


def synth_powered_BH(pvalues_real, pvalues_synth_powered, level, epsilon, method, noise=None):
    """
    synthetic-powered Benjamini-Hochberg procedure.
    """
    p = np.asarray(pvalues_real)
    p_synth = np.asarray(pvalues_synth_powered)
    m = len(p)

    k_vals = np.arange(1, m + 1)
    adj = (k_vals * epsilon / m)[:, None]  # shape (m, 1)

    # Compute \tilde{p}_{k,j} = min{p_j, max{p^synth_j, p_j - k*epsilon/m}}
    P_tilde = np.minimum(p, np.maximum(p_synth, p - adj))  # shape (m, m)
    if noise is not None:
        P_tilde += noise[:, None]
        P_tilde = np.clip(P_tilde, 0, 1)

    # Compute the k-th smallest element for each row (using partition or full sort)
    p_kth = np.partition(P_tilde, k_vals - 1, axis=1)[np.arange(m), k_vals - 1]
    # p_kth = np.sort(P_tilde, axis=1)[np.arange(m), k_vals - 1]

    thresholds = level * k_vals / m
    cond = p_kth <= thresholds

    if np.any(cond):
        k_star = np.max(np.where(cond)[0]) + 1  # +1 for 1-based indexing
        # Recompute \tilde{p} for k_star only
        P_tilde_star = np.minimum(p, np.maximum(p_synth, p - k_star * epsilon / m))
        threshold = np.partition(P_tilde_star, k_star - 1)[k_star - 1]
        rejections = np.where(P_tilde_star <= threshold)[0]
    else:
        k_star, threshold = 0, 0
        rejections = np.array([], dtype=int)

    return rejections, threshold


def get_rejections_indices(calibration_scores, test_scores, level, epsilon=0.01, method='', noise=None):
    if method == 'SynthBH':
        clean_scores, cont_scores = calibration_scores
        cont_clean_scores = np.concatenate([cont_scores, clean_scores], axis=0)
        pvals_real = compute_conformal_pvalues(clean_scores, test_scores)
        pvals_comb = compute_conformal_pvalues(cont_clean_scores, test_scores)
        rejections_indices, threshold = synth_powered_BH(pvals_real, pvals_comb, level, epsilon, method, noise)
    else:
        if method == 'BH_real+e':
            level += epsilon
        # compute conformal p-values
        pvals = compute_conformal_pvalues(calibration_scores, test_scores)
        if noise is not None:
            pvals += noise
            pvals = np.clip(pvals, 0, 1)
        rejections_indices, threshold = BH(pvals, level)
    return rejections_indices, threshold


def get_naive_trimmed_calibration_set(calib_set, calib_y, trim=0.05):
    calib_set_sorted = np.sort(calib_set, axis=0)
    if int(len(calib_set) * trim) > 0:
        model_threshold = calib_set_sorted[-1 * int(len(calib_set) * trim)]
    else:
        model_threshold = np.inf
    our_calib_set = calib_set[calib_set < model_threshold]
    our_calib_y = calib_y[calib_set < model_threshold]
    trimmed_label_samples = calib_y[calib_set >= model_threshold]
    n_trimmed = len(calib_set) - len(our_calib_set)
    return our_calib_set, our_calib_y, trimmed_label_samples, n_trimmed, model_threshold


def get_calibration_set(method, initial_cal, initial_calib_set, calib_set, calib_y, p_trim):
    curr_calib_set, curr_calib_y = None, None
    if method == 'OnlyReal' or method == 'OnlyReal+e':
        curr_calib_set = initial_calib_set
        curr_calib_y = np.zeros(len(initial_calib_set))
    elif method == 'SynthBH':
        _calib_set = calib_set
        _calib_y = calib_y
        curr_calib_set, curr_calib_y, _, _, _ = get_naive_trimmed_calibration_set(
                                                                                    _calib_set,
                                                                                    _calib_y,
                                                                                    trim=p_trim)
        curr_calib_set = (initial_calib_set, curr_calib_set)
        curr_calib_y = (np.zeros(len(initial_calib_set)), curr_calib_y)
    elif method == 'BH_synth':
        _calib_set = calib_set
        _calib_y = calib_y
        curr_calib_set, curr_calib_y, _, _, _ = get_naive_trimmed_calibration_set(
                                                                                    _calib_set,
                                                                                    _calib_y,
                                                                                    trim=p_trim)
    elif method == 'BH_pooled':
        _calib_set = calib_set
        _calib_y = calib_y
        curr_calib_set, curr_calib_y, _, _, _ = get_naive_trimmed_calibration_set(
                                                                                    _calib_set,
                                                                                    _calib_y,
                                                                                    trim=p_trim)
        curr_calib_set = np.concatenate([curr_calib_set, initial_calib_set], axis=0)
        curr_calib_y = np.concatenate([curr_calib_y, np.zeros(len(initial_calib_set))], axis=0)
    return curr_calib_set, curr_calib_y

