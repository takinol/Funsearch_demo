import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
  """返回将物品放入每个箱子的优先级。

参数：

item: 要放入箱子的物品大小。

bins: 每个箱子剩余容量的数组。

返回值：

与 bins 大小相同的数组，包含每个箱子的优先级得分。
  """
  best = -1
  best_score = -np.inf
  scores = np.zeros_like(bins)
  for i in range(len(bins)):
    if bins[i] >= item:
      score = bins[i] / (bins[i] - item + 1e-9)
      scores[i] = score
  return scores

