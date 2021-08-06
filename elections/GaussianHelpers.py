import numpy as np


def cumulative_normal_dist(x_raw: float) -> float:
    if x_raw < 0:
        neg = 1
    else:
        neg = 0

    k = 1.0 / (1.0 + 0.2316419 * abs(x_raw))
    y = ((((1.330274429 * k - 1.821255978) * k + 1.781477937) * k - 0.356563782) * k + 0.319381530) * k
    y = 1.0 - 0.398942280401 * np.exp(-0.5 * (x_raw * x_raw)) * y
    return (1.0 - neg) * y + neg * (1.0 - y)


def rank_to_sigma(rank: float) -> float:
    low = -10.0
    high = 10.0

    if not (1e-5 < rank < 1 - 1e-5):
        print(f"bad rank: {rank}")

    assert(1e-5 < rank < 1 - 1e-5)

    while high - low > 1e-3:
        mid = (high + low) / 2
        s = cumulative_normal_dist(mid)
        if s > rank:
            high = mid
        elif s < rank:
            low = mid
        else:
            return s

    s = (high + low) / 2
    return s
