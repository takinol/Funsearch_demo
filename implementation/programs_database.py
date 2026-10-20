# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""一个实现进化算法的程序数据库。"""

import copy
import dataclasses
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import scipy
from absl import logging
from scipy import cluster
from implementation import code_manipulation
from implementation import config as config_lib

Signature = tuple[float, ...]   #省略号表示任意长度
ScoresPerTest = Mapping[Any, float] #类型别名

# 将得分转换为有偏向性的概率
def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """返回一维有限 `logits` 的带温度 softmax。"""
# 防御性编程
    if not np.all(np.isfinite(logits)): # 检查是否都是有限值
        non_finites = set(logits[~np.isfinite(logits)])
        raise ValueError(f"`logits` contains non-finite value(s): {non_finites}")
    if not np.issubdtype(logits.dtype, np.floating):
        logits = np.array(logits, dtype=np.float32)

    result = scipy.special.softmax(logits / temperature, axis=-1)
    # 确保概率和为1，以防 `np.random.choice` 出错。
    index = np.argmax(result)
    result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index + 1 :])
    return result

# 返回大数据类数据的测试分数
def _reduce_score(scores_per_test: ScoresPerTest) -> float:
    """将每个测试的得分汇总为一个单一得分。"""
    return scores_per_test[list(scores_per_test.keys())[-1]]

# 分数标签（写出元组）
def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
    """将测试得分表示为标准签名。"""
    return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))

# 程序转 prompt 的出口
@dataclasses.dataclass(frozen=True)
class Prompt:
    """由 ProgramsDatabase 生成的提示，用于发送给 Sampler。

    属性:
      code: 提示内容，以待补全函数的函数头结尾。
      version_generated: 待补全的函数为 `_v{version_generated}`。
      island_id: 生成提示中包含的实现的来源岛屿标识符。
                 用于将新生成的实现引导至同一岛屿。
    """

    code: str
    version_generated: int
    island_id: int


class ProgramsDatabase:
    """一个程序集合，按照岛屿组织。"""

    def __init__(
        self,
        config: config_lib.ProgramsDatabaseConfig,
        template: code_manipulation.Program,
        function_to_evolve: str,
    ) -> None:
        self._config: config_lib.ProgramsDatabaseConfig = config
        self._template: code_manipulation.Program = template
        self._function_to_evolve: str = function_to_evolve

        # 初始化空岛屿。
        self._islands: list[Island] = []
        for _ in range(config.num_islands):
            self._islands.append(
                Island(
                    template,
                    function_to_evolve,
                    config.functions_per_prompt,
                    config.cluster_sampling_temperature_init,
                    config.cluster_sampling_temperature_period,
                )
            )
        self._best_score_per_island: list[float] = [-float("inf")] * config.num_islands # 列表乘法
        # 初始化岛屿的得分：负无穷大
        self._best_program_per_island: list[code_manipulation.Function | None] = [None] * config.num_islands
        self._best_scores_per_test_per_island: list[ScoresPerTest | None] = [None] * config.num_islands 
        #最高得分程序的实例的分数

        self._last_reset_time: float = time.time()  #后续岛屿重置会用到的时间

# 返回一个 Prompt 类，给 evaluator
    def get_prompt(self) -> Prompt:
        """返回一个包含来自所选岛屿实现的提示。"""
        island_id = np.random.randint(len(self._islands))
        code, version_generated = self._islands[island_id].get_prompt()     # 告诉要生成的是第几个进化函数
        return Prompt(code, version_generated, island_id)

# 子代存储在父代岛，将程序注册入岛屿，即在岛类添加该程序的信息（被调用）
    def _register_program_in_island(
        self,
        program: code_manipulation.Function,
        island_id: int,
        scores_per_test: ScoresPerTest,
    ) -> None:
        """将 `program` 注册到指定岛屿中。"""
        self._islands[island_id].register_program(program, scores_per_test)
        score = _reduce_score(scores_per_test)
        if score > self._best_score_per_island[island_id]:
            # 存储最佳程序，可能用于后续的岛的重置
            # 三个量：最佳程序，最佳程序的测试数据，最佳的程序得分（已转化）
            self._best_program_per_island[island_id] = program
            self._best_scores_per_test_per_island[island_id] = scores_per_test
            self._best_score_per_island[island_id] = score
            # 标准日志库的输出
            logging.info("Best score of island %d increased to %s", island_id, score)
            print(f"  [岛{island_id}] 新高分: {score:.2f}  |  长度: {len(str(program))} 字符")

# 主流程
    def register_program(
        self,
        program: code_manipulation.Function,
        island_id: int | None,
        scores_per_test: ScoresPerTest,
    ) -> None:
        """将 `program` 注册到数据库中。"""
        # 在异步实现中，我们应考虑在提示生成后岛屿被重置但仍尝试注册程序的情况。
        # 为了简单，此处没有处理该情况。
        if island_id is None:
            # 这是开始时添加的程序，因此将其添加到所有岛屿。
            for island_id in range(len(self._islands)):
                self._register_program_in_island(program, island_id, scores_per_test)
        else:
            self._register_program_in_island(program, island_id, scores_per_test)

        # 检查是否到了重置岛屿的时间。
        if time.time() - self._last_reset_time > self._config.reset_period:
            self._last_reset_time = time.time()
            self.reset_islands()

# 重置岛（被调用）
    def reset_islands(self) -> None:
        """重置较弱的一半岛屿。"""
        # 我们在最佳分数上加入少量噪声后排序，以打破平局。
        indices_sorted_by_score: np.ndarray = np.argsort(
            self._best_score_per_island + np.random.randn(len(self._best_score_per_island)) * 1e-6
        )
        # 一半的岛被重置
        num_islands_to_reset = self._config.num_islands // 2
        reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
        keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
        for island_id in reset_islands_ids:
            self._islands[island_id] = Island(
                self._template,
                self._function_to_evolve,
                self._config.functions_per_prompt,
                self._config.cluster_sampling_temperature_init,
                self._config.cluster_sampling_temperature_period,
            )
            self._best_score_per_island[island_id] = -float("inf")  # 避免无法覆盖原分数
            founder_island_id = np.random.choice(keep_islands_ids)  # 随机选一个50%优秀的岛屿
            founder = self._best_program_per_island[founder_island_id]  #拿该岛的最佳程序
            founder_scores = self._best_scores_per_test_per_island[founder_island_id]
            self._register_program_in_island(founder, island_id, founder_scores)

# 输出数据的接口————这个是你自己写的对外接口【可视化】
    def get_island_states(self)->list[dict]:
        """返回每个岛屿的状态摘要"""
        states=[]
        for i,island in enumerate(self._islands):
            states.append({
                "island_id": i,
                "num_programs": island._num_programs,
                "num_clusters": len(island._clusters),
                "best_score": self._best_score_per_island[i],
            })
        return states

    def export_all_programs(self):
        programs=[]
        for i,island in enumerate(self._islands):
            for sig,cluster in island._clusters.items():
                for prog in cluster._programs:
                    programs.append({
                    "island_id": i,
                    "signature": list(sig),
                    "score": cluster.score,
                    "code": str(prog),
                    "length": len(str(prog)),
                })
        return programs


class Island:
    """程序数据库中的一个子种群。"""

    def __init__(
        self,
        template: code_manipulation.Program,
        function_to_evolve: str,
        functions_per_prompt: int,
        cluster_sampling_temperature_init: float,
        cluster_sampling_temperature_period: int,
    ) -> None:
        self._template: code_manipulation.Program = template
        self._function_to_evolve: str = function_to_evolve
        self._functions_per_prompt: int = functions_per_prompt
        self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self._cluster_sampling_temperature_period = cluster_sampling_temperature_period

        self._clusters: dict[Signature, Cluster] = {}
        self._num_programs: int = 0

# 将标签一致的程序聚合到同一个簇，程序进库
    def register_program(
        self,
        program: code_manipulation.Function,
        scores_per_test: ScoresPerTest,
    ) -> None:
        """将程序存储到该岛屿中对应的簇内。"""
        signature = _get_signature(scores_per_test)
        if signature not in self._clusters:
            # 簇的得分
            score = _reduce_score(scores_per_test)
            self._clusters[signature] = Cluster(score, program)
        else:
            self._clusters[signature].register_program(program)
        self._num_programs += 1

# 选择簇并选择程序，调用 _generate_prompt 来返回 prompt
    def get_prompt(self) -> tuple[str, int]:
        """构造一个包含该岛屿中函数的提示。"""
        signatures = list(self._clusters.keys())
        # 这里的 score 是取大数据测试的分数
        cluster_scores = np.array([self._clusters[signature].score for signature in signatures])   # 这里用上了装饰器

        # 使用带温度调度的 softmax 将分数转换为概率。
        period = self._cluster_sampling_temperature_period
        # 调控温度，高温（解决0）更随机，低温几乎只选高分簇
        temperature = self._cluster_sampling_temperature_init * (1 - (self._num_programs % period) / period)
        #周期性退火（Cyclical Annealing）” 或 “锯齿波衰减（Sawtooth Decay）” 策略

        probabilities = _softmax(cluster_scores, temperature)   # 得到概论分布

        # 在实验初期簇较少时，将更少的程序放入提示中。
        functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)

        # 选择簇
        idx = np.random.choice(len(signatures), size=functions_per_prompt, p=probabilities,replace=False)

        chosen_signatures = [signatures[i] for i in idx]        # 其实这里就发生了簇的交叉，进而给到 LLMs
        # 候选程序集中得地方
        implementations = []
        scores = []
        # 抽出来，要做成 prompt 去输出
        for signature in chosen_signatures:
            cluster = self._clusters[signature]
            # 在簇里面选择程序（倾向于短程序）
            implementations.append(cluster.sample_program())
            #这里你就可以发现其实是从不同簇抽出我们需要的数量的程序
            # 簇的得分也是该程序的得分（跑测试案例的得分）
            scores.append(cluster.score)

# 到现在，implementation 里面装着所有从不同簇里面得到的一个的程序，且排好了顺序
        indices = np.argsort(scores)
        sorted_implementations = [implementations[i] for i in indices]
        # 和 functions_per_prompt 有关，告诉你这次会产生多少个新的程序
        version_generated = len(sorted_implementations) + 1
        return self._generate_prompt(sorted_implementations), version_generated

    def _generate_prompt(self, implementations: Sequence[code_manipulation.Function]) -> str:
        """创建一个包含一系列函数 `implementations` 的提示。"""
        implementations = copy.deepcopy(implementations)  # 我们将对这些副本进行修改。

        # 格式化要包含在提示中的函数名称和文档字符串。
        versioned_functions: list[code_manipulation.Function] = []
        for i, implementation in enumerate(implementations):
            # 组装新的函数头
            new_function_name = f"{self._function_to_evolve}_v{i}"
            # 更新函数头
            implementation.name = new_function_name #开始给到新的名字
            # 更新 `_v0` 之后所有后续函数的文档字符串。
            if i >= 1:
                #给予提示词
                implementation.docstring = f"Improved version of `{self._function_to_evolve}_v{i - 1}`."
                #给点进化的提示词
            # 如果函数是递归的，则将对自身的调用替换为新名称。
            # 查找独立的函数调用，add 改为 add_v1，且这个程序变成字符串
            implementation = code_manipulation.rename_function_calls(
                str(implementation), self._function_to_evolve, new_function_name
            )
            # versioned_functions 是最终改好函数头的 function 列表
            versioned_functions.append(code_manipulation.text_to_function(implementation))

        # 创建由 LLM 生成的函数头。
        # 留一个空位给 LLM
        next_version = len(implementations)
        new_function_name = f"{self._function_to_evolve}_v{next_version}"
# dataclasses.replace ，基于一个现有数据类实例，创建一个新的实例，同时只修改你指定的字段
        header = dataclasses.replace(
            implementations[-1],
            name=new_function_name,
            body="",
            docstring=(f"Improved version of `{self._function_to_evolve}_v{next_version - 1}`."),
        )
        versioned_functions.append(header)

        # 用此处构造的函数列表替换模板中的函数。
        prompt = dataclasses.replace(self._template, functions=versioned_functions)
        return str(prompt)


class Cluster:
    """同一岛屿上具有相同 Signature 的程序簇。"""

    def __init__(self, score: float, implementation: code_manipulation.Function):
        self._score = score
        self._programs: list[code_manipulation.Function] = [implementation]
        self._lengths: list[int] = [len(str(implementation))]

    @property
    def score(self) -> float:
        """该簇所代表签名的汇总得分。"""
        return self._score

# 程序入口
    def register_program(self, program: code_manipulation.Function) -> None:
        """将 `program` 添加到该簇中。"""
        self._programs.append(program)
        self._lengths.append(len(str(program)))

# 程序出口，偏好短代码
    def sample_program(self) -> code_manipulation.Function:
        """采样一个程序，较短的程序获得更高的概率。"""
        normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (max(self._lengths) + 1e-6)
        probabilities = _softmax(-normalized_lengths, temperature=1.0)
        idx = np.random.choice(len(self._programs), size=1, p=probabilities)
        return self._programs[int(idx[0])]          # 改法 1
