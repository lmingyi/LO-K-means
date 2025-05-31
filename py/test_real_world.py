"""
Compare K-means, C-LO_K_means, D-LO_K_means, and Min_D_LO_K_means by loss, time, and steps.
"""

import time
import numpy as np
from LO_K_means import LO_K_means


def make_unique(X: np.ndarray):
    unique_X, weight = np.unique(X, axis=0, return_counts=True)
    return unique_X, weight


def compute_stats(arr):
    a = np.array(arr, dtype=float)
    return a.mean(), a.std()


def print_stats(name, losses, times, steps):
    loss_avg, loss_std = compute_stats(losses)
    loss_min = np.min(losses)
    time_avg = np.mean(times)
    step_avg = np.mean(steps)
    print(f"{name}:")
    print(f" Loss (avg ± std):  {loss_avg:.4f} ± {loss_std:.4f}")
    print(f" Loss (min):        {loss_min:.4f}")
    print(f" Time (avg):        {time_avg:.4f} sec")
    print(f" Steps (avg):       {step_avg:.1f}")
    print()


def main():
    # Read dataset
    with open("../data/Iris.txt") as f:
        N, D = map(int, f.readline().split())
        data = np.loadtxt(f, delimiter=None).reshape(N, D)

    X_unique, weight = make_unique(data)
    K = 50
    eps = 1e-10

    k_loss, k_time, k_step = [], [], []
    c_loss, c_time, c_step = [], [], []
    d_loss, d_time, d_step = [], [], []
    m_loss, m_time, m_step = [], [], []

    for _ in range(20):

        # K-means setting
        kmeans = LO_K_means(
            X=X_unique,
            weight=weight,
            K=K,
            init_="rand",
            breg="KL",
            random_state=None,
            eps=eps,
        )

        # Standard K-means
        t0 = time.perf_counter()
        assign_k, centers_k, loss_k = kmeans.K_means()
        t1 = time.perf_counter()
        k_loss.append(loss_k)
        k_time.append(t1 - t0)
        k_step.append(kmeans.step_num)

        # C-LO_K_means (Function 1)
        t0 = time.perf_counter()
        assign_c, centers_c, loss_c = kmeans.C_LO_K_means()
        t1 = time.perf_counter()
        c_loss.append(loss_c)
        c_time.append(t1 - t0)
        c_step.append(kmeans.step_num)

        # D-LO_K_means (Function 2)
        t0 = time.perf_counter()
        assign_d, centers_d, loss_d = kmeans.D_LO_K_means()
        t1 = time.perf_counter()
        d_loss.append(loss_d)
        d_time.append(t1 - t0)
        d_step.append(kmeans.step_num)

        # Min-D-LO_K_means (Function 3)
        t0 = time.perf_counter()
        assign_m, centers_m, loss_m = kmeans.Min_D_LO_K_means()
        t1 = time.perf_counter()
        m_loss.append(loss_m)
        m_time.append(t1 - t0)
        m_step.append(kmeans.step_num)

    # Print summary statistics
    print_stats("K-means", k_loss, k_time, k_step)
    print_stats("C-LO_K_means", c_loss, c_time, c_step)
    print_stats("D-LO_K_means", d_loss, d_time, d_step)
    print_stats("Min_D_LO_K_means", m_loss, m_time, m_step)


if __name__ == "__main__":
    main()
