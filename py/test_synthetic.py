"""
Synthetic dataset experiment: Result in "./synthetic_result/".
Compute clustering loss over [min_range, max_range]^n_features,
"""

import numpy as np
import pandas as pd
from LO_K_means import LO_K_means

# settings
n_features = 2
min_range = 1
max_range = 10
times = 1000
n_samples_range = range(20, 110, 10)
n_clusters_range = lambda n: range(2, min(31, n - 4), 4)

eps = 1e-10
rng = np.random.default_rng()


def make_unique(X):
    unique_X, weight = np.unique(X, axis=0, return_counts=True)
    return unique_X, weight.astype(float)


# Run K-means and D-LO-K-means
def calc_k_means_discrete(X, weight, n_clusters):
    if len(X) <= n_clusters:
        return 0.0, 0.0, 0, 0, 0

    # K-means setting
    kmeans = LO_K_means(
        X=X,
        weight=weight,
        K=n_clusters,
        init_="rand",
        breg="squared",
        random_state=None,
        eps=eps,
    )

    assign_o, centers_o, loss_o = kmeans.K_means()
    step_o = kmeans.step_num
    assign_d, centers_d, loss_d = kmeans.D_LO_K_means()
    step_d, imp_d = kmeans.step_num, kmeans.imp_num

    return loss_o, loss_d, step_o, step_d, imp_d


def calctimes_discrete(n_samples, n_clusters, times):
    cnt = 0
    ord_val, mod_val, ord_roop, mod_roop, imp_roop = [], [], [], [], []
    while cnt < times:
        X = rng.integers(min_range, max_range + 1, size=(n_samples, n_features))
        X_u, weight = make_unique(X)
        ord_loss, mod_loss, ord_r, mod_r, imp_r = calc_k_means_discrete(X_u, weight, n_clusters)
        cnt += 1
        ord_val.append(ord_loss)
        mod_val.append(mod_loss)
        ord_roop.append(ord_r)
        mod_roop.append(mod_r)
        imp_roop.append(imp_r)
    return ord_val, mod_val, ord_roop, mod_roop, imp_roop


def save_results(output_file):
    with open(output_file, "w") as f:
        for n_samples in n_samples_range:
            for n_clusters in n_clusters_range(n_samples):
                print(n_samples, n_clusters)
                ord_val, mod_val, ord_roop, mod_roop, imp_roop = calctimes_discrete(n_samples, n_clusters, times)

                f.write(f"n_samples = {n_samples}, n_clusters = {n_clusters}\n")
                f.write(f"ord_val : {ord_val}\n")
                f.write(f"mod_val : {mod_val}\n")
                f.write(f"ord_roop : {ord_roop}\n")
                f.write(f"mod_roop : {mod_roop}\n")
                f.write(f"imp_roop : {imp_roop}\n\n")


if __name__ == "__main__":
    save_results("./synthetic_result/K-means_discrete_2d_1-10_1000times.txt")
