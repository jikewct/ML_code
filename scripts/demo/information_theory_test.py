import scipy.special
import scipy.stats
import torch
import math
import scipy
import numpy as np
from entropy_estimators import continuous



def test_entropy_estimate():

    dim = 50
    sample_num = 1000
    k = 1
    exact_entropy = dim / 2.0 * ( math.log(2 * math.pi) + 1 )
    scipy_entropy = scipy.stats.multivariate_normal.entropy(np.zeros([dim]), np.eye(dim))
    samples = torch.randn([sample_num, dim])
    knn_l2_entropy = continuous.get_h(samples, norm='euclidean', k = k);
    knn_max_entropy = continuous.get_h(samples, norm='max', k = k);
    samples_rows = torch.unsqueeze(samples, 0)
    samples_cols = torch.unsqueeze(samples, 1)

    gram_l2 = torch.sqrt(torch.sum((samples_cols - samples_rows) ** 2, dim = -1)) + torch.eye(sample_num) * 10000
    min_l2_vector, _ = torch.min(gram_l2, dim = -1)
    log_sum_min = torch.log(min_l2_vector).sum()
    #n dimension unit ball volumn
    log_volumn = math.log( math.pi ** (dim /2.0) / scipy.special.gamma(dim /2 + 1))
    approx_l2_entroy = dim /float(sample_num) * log_sum_min + log_volumn + scipy.special.psi(sample_num) -  scipy.special.psi(1)
    log_sum_min_max = torch.log(torch.min(torch.max((samples_cols - samples_rows).abs(), dim = -1)[0] + torch.eye(sample_num) * 10000, dim = -1)[0]).sum()
    approx_max_entroy = dim /float(sample_num) * log_sum_min_max  + math.log(2**dim) + scipy.special.psi(sample_num) -  scipy.special.psi(1)

    error_rate = ((exact_entropy - approx_l2_entroy)/exact_entropy).abs()
    print("scipy_value:{:.8f}, exact_value:{:.8f}, approx_value:{:.8f},approx_max_entroy:{:.8f}, knn_l2_entropy:{:.8f},knn_max_entropy:{:.8f}, error_rate:{:.8f}".format(
            scipy_entropy, exact_entropy, approx_l2_entroy,approx_max_entroy, knn_l2_entropy,knn_max_entropy, error_rate
        ))




if __name__ == "__main__":
    test_entropy_estimate()