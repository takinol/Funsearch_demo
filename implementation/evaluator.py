# 版权所有 2023 DeepMind Technologies Limited
#
# 根据 Apache License, Version 2.0（“许可证”）授权；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下位置获取许可证副本：
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发的软件
# 没有任何明示或暗示的保证或条件。
# 请参阅许可证了解具体语言 governing permissions and
# limitations under the License.
# ==============================================================================

"""用于评估 Sampler 提出的程序的类。"""
import ast
from collections.abc import Sequence
import copy
from typing import Any

from implementation import code_manipulation
from implementation import programs_database


class _FunctionLineVisitor(ast.NodeVisitor):
  """访问器，用于查找具有给定名称的函数的最后一行号。"""

  def __init__(self, target_function_name: str) -> None:
    self._target_function_name: str = target_function_name
    self._function_end_line: int | None = None

  def visit_FunctionDef(self, node: Any) -> None:  # pylint: disable=invalid-name
    """收集目标函数的结束行号。"""
    if node.name == self._target_function_name:
      self._function_end_line = node.end_lineno
    self.generic_visit(node)

  @property
  def function_end_line(self) -> int:
    """函数 `target_function_name` 最后一行的行号。"""
    assert self._function_end_line is not None  # 检查内部正确性。
    return self._function_end_line


def _strip_markdown_fences(code: str) -> str:
  """去掉 LLM 偶尔输出的 ``` / ```python 围栏。"""
  lines = code.splitlines()
  while lines and lines[0].strip().startswith('```'):
    lines.pop(0)
  while lines and lines[-1].strip().startswith('```'):
    lines.pop()
  return '\n'.join(lines)


def _extract_standalone_function_body(code: str) -> str | None:
  """如果 LLM 输出了完整 def，则提取其中的函数体。"""
  try:
    tree = ast.parse(code)
  except SyntaxError:
    return None

  for node in tree.body:
    if not isinstance(node, ast.FunctionDef) or not node.body:
      continue
    code_lines = code.splitlines()
    start_line = node.body[0].lineno - 1

    # 跳过函数自带的 docstring，模板里已经有一个 docstring 了。
    first = node.body[0]
    if (isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)):
      if len(node.body) > 1:
        start_line = node.body[1].lineno - 1
      else:
        start_line = node.end_lineno

    return '\n'.join(code_lines[start_line:node.end_lineno])
  return None


def _ensure_indented_body(code: str) -> str:
  """把 LLM 输出恢复成可插入函数体的缩进形式。"""
  lines = code.splitlines()

  def _code_indent(line: str) -> int | None:
    """返回代码行的缩进；空行和注释返回 None。"""
    stripped = line.strip()
    if not stripped or stripped.startswith('#'):
      return None
    return len(line) - len(line.lstrip())

  indents = [_code_indent(line) for line in lines]
  code_indices = [i for i, v in enumerate(indents) if v is not None]
  if not code_indices:
    return code

  first_code = code_indices[0]
  # 第一行已经有缩进，说明 body 基本正常，直接返回。
  if indents[first_code] > 0:
    return code

  following = [indents[i] for i in code_indices[1:]]
  if following and min(following) > 0:
    # 常见情况：只有第一行丢了缩进，其余行已经带函数体缩进。
    # 只给第一行补两格，避免把后面的同级代码变成“意外缩进”。
    lines[first_code] = '  ' + lines[first_code]
    return '\n'.join(lines)

  # 所有代码行都顶格：整体补两格，保持相对缩进不变。
  return '\n'.join(('  ' + line) if line.strip() else line for line in lines)


def _normalize_generated_code(generated_code: str) -> str:
  """把 LLM 输出整理成可插入函数体的代码。"""
  generated_code = _strip_markdown_fences(generated_code).strip('\n')
  body = _extract_standalone_function_body(generated_code)
  if body is not None:
    return body
  return _ensure_indented_body(generated_code)


def _trim_function_body(generated_code: str) -> str:
  """提取生成的函数的主体，并修剪其后的所有内容。"""
  generated_code = _normalize_generated_code(generated_code or '')
  if not generated_code:
    return ''
  code = f'def fake_function_header():\n{generated_code}'
  tree = None
  # 我们持续尝试并从末尾删除代码，直到解析器成功。
  while tree is None:
    try:
      tree = ast.parse(code)
    except SyntaxError as e:
     code = '\n'.join(code.splitlines()[:e.lineno - 1]) # type: ignore
  if not code:
    # 无法从 `generated_code` 中保存任何内容
    return ''

  visitor = _FunctionLineVisitor('fake_function_header')
  visitor.visit(tree)
  body_lines = code.splitlines()[1:visitor.function_end_line]
  return '\n'.join(body_lines) + '\n\n'


def _sample_to_program(
    generated_code: str,
    version_generated: int | None,
    template: code_manipulation.Program,
    function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:
  """返回编译后的生成函数和完整的可运行程序。"""
  body = _trim_function_body(generated_code)    # 得到缩进好换行好的代码str
  if version_generated is not None:
    body = code_manipulation.rename_function_calls( #黑盒
        body,
        f'{function_to_evolve}_v{version_generated}',
        function_to_evolve)

  program = copy.deepcopy(template)
  evolved_function = program.get_function(function_to_evolve)
  evolved_function.body = body  #换成新生成的函数
  return evolved_function, str(program)


class Sandbox:
  """用于执行生成代码的沙盒。"""

  def run(
      self,
      program: str,
      function_to_run: str,
      test_input: str,
      timeout_seconds: int,
  ) -> tuple[Any, bool]:
    """返回 `function_to_run(test_input)` 及执行是否成功。"""
    raise NotImplementedError(
        'Must provide a sandbox for executing untrusted code.')


def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
  """返回生成的函数是否调用了早期版本。"""
  for name in code_manipulation.get_functions_called(program):
    # 在传入此函数的 `program` 中，最新生成的函数已被重命名为
    # `function_to_evolve`（不带后缀）。因此，任何以 `function_to_evolve_v`
    # 开头的函数调用都是对祖先函数的调用。
    
    if name.startswith(f'{function_to_evolve}_v'):
      return True
  return False


class Evaluator:
  """分析 LLM 生成的函数的类。"""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      template: code_manipulation.Program,
      function_to_evolve: str,
      function_to_run: str,
      inputs: Sequence[Any],  # 数据集名字的序列
      sandbox: Sandbox,
      timeout_seconds: int = 30
  ):
    self._database = database
    self._template = template
    self._function_to_evolve = function_to_evolve
    self._function_to_run = function_to_run
    self._inputs = inputs
    self._timeout_seconds = timeout_seconds
    self._sandbox = sandbox

  # def analyse(
  #     self,
  #     sample: str,
  #     island_id: int | None,
  #     version_generated: int | None,
  # ) -> None:
  #   """将样本编译成程序并在测试输入上执行。"""
    
  #   new_function, program = _sample_to_program(
  #       sample, version_generated, self._template, self._function_to_evolve)

  #   scores_per_test = {}
  #   for current_input in self._inputs:#一个个取数据集的名称
  #     test_output, runs_ok = self._sandbox.run(
  #         program, self._function_to_run, current_input, self._timeout_seconds)

  #     if (runs_ok and not _calls_ancestor(program, self._function_to_evolve)
  #         and test_output is not None):
  #       if not isinstance(test_output, (int, float)):
  #         raise ValueError('@function.run 未返回 int/float 分数。')# 简单检测一下数据类型

  #       scores_per_test[current_input] = test_output
  #   if scores_per_test:
  #     self._database.register_program(new_function, island_id, scores_per_test)
  #     #实现通过检测的函数入库
  def analyse(
    self,
    sample: str,
    island_id: int | None,
    version_generated: int | None,
) -> dict:
    """评估一个样本，返回结构化记录；入库与否看 accepted。"""

    # 先保证任何分支都返回一条可写日志的记录
    record = {
        "accepted": False,
        "reason": None,
        "scores_per_test": {},
        "body": "",
        "code": "",
    }

    try:
        new_function, program = _sample_to_program(
            sample, version_generated, self._template, self._function_to_evolve)
    except Exception as e:  # noqa: BLE001
        record["reason"] = f"parse_error:{type(e).__name__}:{e}"
        record["code"] = sample
        return record

    body = new_function.body or ""
    record["body"] = body
    record["code"] = str(new_function)

    # LLM 输出了空函数体，直接判失败，不用跑沙箱
    if not body.strip():
        record["reason"] = "empty_body"
        return record

    scores_per_test = {}
    failures = []

    for current_input in self._inputs:
        test_output, runs_ok = self._sandbox.run(
            program, self._function_to_run, current_input,
            self._timeout_seconds)

        if not runs_ok:
            failures.append(f"{current_input}:runtime_error")
            continue
        if _calls_ancestor(program, self._function_to_evolve):
            failures.append(f"{current_input}:calls_ancestor")
            continue
        if test_output is None:
            failures.append(f"{current_input}:invalid_output")
            continue
        if not isinstance(test_output, (int, float)):
            failures.append(f"{current_input}:bad_type")
            continue

        scores_per_test[current_input] = test_output

    record["scores_per_test"] = scores_per_test

    # 我建议：所有测试集都通过才算 accepted。
    # 现在的代码是“只要有一个分数就入库”，会产生签名不完整的程序。
    if len(scores_per_test) == len(self._inputs) and not failures:
        self._database.register_program(
            new_function, island_id, scores_per_test)
        record["accepted"] = True
    else:
        record["reason"] = ";".join(failures) or "partial_scores"

    return record
