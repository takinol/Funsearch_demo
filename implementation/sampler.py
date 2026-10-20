# 版权所有 2023 DeepMind Technologies Limited
#
# 根据 Apache License, Version 2.0（“许可证”）授权；
# 您不得使用此文件，除非遵守许可证。
# 您可以在以下网址获取许可证副本：
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发的软件
# 没有任何明示或暗示的保证或条件。
# 请参阅许可证以了解具体的语言 governing permissions and
# limitations under the License.
# ==============================================================================

"""用于采样新程序的类。"""
import hashlib
import json
import os
from collections.abc import Collection, Sequence

import numpy as np

from implementation import evaluator, programs_database


class LLM:
  """预测所提供的源代码续写的语言模型。"""

  def __init__(self, samples_per_prompt: int) -> None:
    self._samples_per_prompt = samples_per_prompt

  def _draw_sample(self, prompt: str) -> str:
    """返回 `prompt` 的预测续写。"""
    raise NotImplementedError('必须提供一个语言模型。')

  def draw_samples(self, prompt: str) -> Collection[str]:
    """返回 `prompt` 的多个预测续写。"""
    return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]


class Sampler:
  """负责采样程序续写并将其发送进行分析的节点。"""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      evaluators: Sequence[evaluator.Evaluator],
      samples_per_prompt: int,
      llm:LLM
  ) -> None:
    self._database = database
    self._evaluators = evaluators
    self._llm = llm

    #新增日志初始化
    self._results_dir = "results"
    os.makedirs(self._results_dir, exist_ok=True)
    self._samples_path = os.path.join(self._results_dir, "samples.jsonl")
    self._metrics_path = os.path.join(self._results_dir, "metrics.jsonl")
    self._best_path = os.path.join(self._results_dir, "best_program.py")
    self._best_seen = -float("inf")#存储着最佳分数
    self._seen_prompt_hashes = set()

  def sample(self,max_iterations:int):
    """持续获取提示，采样程序，并将其发送进行分析。"""
    iterations=0    #轮次计数器

    try:
        while max_iterations-iterations >0:
            prompt = self._database.get_prompt()
            samples = self._llm.draw_samples(prompt.code)
              # 此循环可在远程评估器机器上并行执行。
            # for sample in samples:
            #     chosen_evaluator_idx = np.random.choice(len(self._evaluators))
            #     chosen_evaluator=self._evaluators[chosen_evaluator_idx]
            #     chosen_evaluator.analyse(
            #         sample, prompt.island_id, prompt.version_generated)

            prompt_hash = hashlib.sha256(
            prompt.code.encode("utf-8")).hexdigest()[:16]

            # 每个唯一 prompt 只记录一次，供 samples 关联父代
            if prompt_hash not in self._seen_prompt_hashes:
                self._seen_prompt_hashes.add(prompt_hash)
                self._log_prompt(iterations, prompt)

            for sample_idx, sample in enumerate(samples):
              chosen_evaluator = self._evaluators[
                  np.random.choice(len(self._evaluators))]
              result = chosen_evaluator.analyse(
                  sample, prompt.island_id, prompt.version_generated)
              self._log_sample(iterations, sample_idx, prompt_hash,
                     prompt, sample, result)

            #新增
            states=self._database.get_island_states()
            global_best_score=max(s['best_score'] for s in states)
            record={
                "iteration":iterations,
                "global_best_score":global_best_score,
                "island_bests":[s['best_score'] for s in states],
                "num_programs":sum(s["num_programs"]for s in states),
                "num_clusters":sum(s["num_clusters"]for s in states)
            }
            self._log_metrics(record)

                    # [新增] 只有破纪录时才保存最佳程序（避免每轮重写文件）
            if global_best_score > self._best_seen:
                self._best_seen = global_best_score
                self._save_best_program()
            print(f"[iter {iterations}] best={global_best_score:.2f} "
                        f"programs={record['num_programs']}")   # [新增] 控制台状态行
            iterations+=1

    except KeyboardInterrupt :
        print("外界输入中断信号")

      # ===== [新增] 两个辅助方法 =====
  def _log_metrics(self, record: dict) -> None:
        """把一轮的指标追加进 metrics.jsonl（每行一条 JSON）。"""
        with open(self._metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

  def _log_sample(self, iteration, sample_idx, prompt_hash,
                prompt, raw_output, result) -> None:
    """把一次 LLM 样本的评估结果追加进 samples.jsonl。"""

    record = {
        "iteration": iteration,
        "sample_idx": sample_idx,
        "island_id": prompt.island_id,
        "version_generated": prompt.version_generated,
        "prompt_hash": prompt_hash,
        "accepted": bool(result and result.get("accepted")),
        "reason": (result or {}).get("reason"),
        "raw_output": raw_output,
    }

    if result:
        scores = result.get("scores_per_test") or {}
        if scores:
            # 与 programs_database._reduce_score 保持一致：取最后一个数据集
            record["signature"] = [scores[k] for k in sorted(scores)]
            record["score"] = list(scores.values())[-1]

        record["body"] = result.get("body", "")
        record["code"] = result.get("code", "")
        record["code_hash"] = hashlib.sha256(
            record["code"].encode("utf-8")).hexdigest()[:16]

    with open(self._samples_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

  def _log_prompt(self, iteration, prompt) -> None:
    """每个唯一 prompt 只记录一次，供 samples 关联父代。"""
    prompt_hash = hashlib.sha256(
        prompt.code.encode("utf-8")).hexdigest()[:16]
    with open(os.path.join(self._results_dir, "prompts.jsonl"),
              "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "iteration": iteration,
            "prompt_hash": prompt_hash,
            "island_id": prompt.island_id,
            "version_generated": prompt.version_generated,
            "prompt": prompt.code,
        }, ensure_ascii=False) + "\n")

#上面为新增

  def _save_best_program(self) -> None:
        """把当前全局最高分程序写进 best_program.py。"""
        programs = self._database.export_all_programs()   # 用你自己加的方法
        if not programs:
            return
        best = max(programs, key=lambda p: p["score"])
        with open(self._best_path, "w", encoding="utf-8") as f:
            f.write(best["code"])
