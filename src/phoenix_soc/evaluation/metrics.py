from typing import List, Tuple

def precision_recall(tp: int, fp: int, fn: int) -> Tuple[float, float]:
    p = tp / (tp + fp) if (tp+fp) else 0.0
    r = tp / (tp + fn) if (tp+fn) else 0.0
    return p, r

# mean time to detect / respond (mock)
def mttd_mttr(detect_secs: List[int], respond_secs: List[int]) -> Tuple[float, float]:
    avg = lambda xs: sum(xs)/len(xs) if xs else 0.0
    return avg(detect_secs), avg(respond_secs)
