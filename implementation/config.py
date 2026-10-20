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

"""FunSearch实验的配置"""
import dataclasses


@dataclasses.dataclass(frozen=True)
class ProgramsDatabaseConfig:
  """ProgramsDatabase的配置.

属性:
- functions_per_prompt: 提示(prompt)中包含的过往程序数量.
- num_islands: 作为多样性机制维护的岛屿(island)数量.
- reset_period: 重置最弱岛屿的时间间隔(单位:秒).
- cluster_sampling_temperature_init: 对岛屿内簇(cluster)进行Softmax采样的初始温度.
- cluster_sampling_temperature_period: 簇采样温度的线性衰减周期.
  """
  functions_per_prompt: int = 2
  num_islands: int = 10
  reset_period: int = 4 * 60 * 60
  cluster_sampling_temperature_init: float = 0.1
  cluster_sampling_temperature_period: int = 30_000


@dataclasses.dataclass(frozen=True)
class Config:
  """FunSearch实验的配置.

属性:
- programs_database: 进化算法的配置.
- num_samplers: 实验中独立的采样器(Sampler)数量. 大于1的值仅在采样器能够并行执行时有效，例如在分布式系统的不同机器上.
- num_evaluators: 实验中独立的程序评估器(Evaluator)数量. 大于1的值仅在评估器能够作为分布式系统的一部分并行执行时才有用.
- samples_per_prompt: 每个提示(prompt)应独立采样获得的程序续写(program continuation)数量.
  """
  programs_database: ProgramsDatabaseConfig = dataclasses.field(
      default_factory=ProgramsDatabaseConfig)
  num_samplers: int = 15
  num_evaluators: int = 140
  samples_per_prompt: int = 4


