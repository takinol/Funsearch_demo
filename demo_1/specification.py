specification='''
import numpy as np

from implementation import evaluator
"""寻找一维在线装箱问题的启发式算法。"""

def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
  """返回物品可以放入的箱子的索引。"""
  return np.nonzero((bins - item) >= 0)[0]


def online_binpack(
    items: tuple[float, ...], bins: np.ndarray
) -> tuple[list[list[float]], np.ndarray]:
  """Performs online binpacking of `items` into `bins`.

  将 `items` 在线装入 `bins` 的装箱过程。
  """
  # 跟踪每个箱子中加入了哪些物品。
  packing = [[] for _ in bins]
  # 将物品放入箱子中。
  for item in items:
    # 找出有足够剩余容量容纳该物品的箱子。
    valid_bin_indices = get_valid_bin_indices(item, bins)
    # 根据启发式规则为每个箱子打分。
    candidate_bins = bins[valid_bin_indices]
    priorities = priority(item, candidate_bins)
    # 校验返回值：残缺/无 return 的函数会返回 None，np.argmax(None)=0
    # 会被静默当成 First Fit 接受，这里必须显式拦截。
    priority_array = np.asarray(priorities)
    if (priority_array.shape != candidate_bins.shape
        or not np.issubdtype(priority_array.dtype, np.number)):
      raise ValueError(
          'priority() must return one numeric score per candidate bin')
    # 将物品放入优先级最高的箱子中。
    best_bin = valid_bin_indices[np.argmax(priority_array)]
    bins[best_bin] -= item
    packing[best_bin].append(item)
  # 从装箱结果中移除未使用的箱子。
  packing = [bin_items for bin_items in packing if bin_items]
  return packing, bins


# @funsearch.run
def evaluate(dataset_name : str) -> float:
  """Evaluate heuristic function on a set of online binpacking instances.

  对一组在线装箱实例上的启发式函数进行评估。
  """
  # 存储每个实例使用箱子数量的列表。
  num_bins = []
  # 对每个实例执行在线装箱。
  instances=DATASET[dataset_name]

  for name in instances:
    instance = instances[name]
    capacity = instance['capacity']
    items = instance['items']
    # 创建 num_items 个箱子，以保证无论装箱顺序如何，都始终有足够空间容纳所有物品。
    # 数组形状为 (num_items,)。
    bins = np.array([capacity for _ in range(instance['num_items'])])
    # 将物品装入箱子，并返回装箱后的剩余容量 bins_packed，其形状为 (num_items,)。
    _, bins_packed = online_binpack(items, bins)
    # 如果某个箱子的剩余容量等于初始容量，则说明它未被使用；统计已使用的箱子数量。
    num_bins.append((bins_packed != capacity).sum())
  # 启发式函数的评分是各实例所用箱子数量平均值的负值，因为我们希望最小化箱子数量。
  return -float(np.mean(num_bins))


# @funsearch.evolve
def priority(item: float, bins: np.ndarray) -> np.ndarray:
  """返回将物品放入每个箱子的优先级。

参数：

item: 要放入箱子的物品大小。

bins: 每个箱子剩余容量的数组。

返回值：

与 bins 大小相同的数组，包含每个箱子的优先级得分。
  """
  return -(bins - item)
'''
