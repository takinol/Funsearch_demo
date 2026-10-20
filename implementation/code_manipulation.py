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

"""用于操作 Python 代码的工具。

它实现了两个表示代码单元的类：

Function（函数）：包含我们需要了解的关于函数的所有信息，包括名称、参数、函数体，以及可选的返回值类型和文档字符串。

Program（程序）：包含代码前言（可以是导入语句、全局变量和类等）以及一个 Function 列表。
"""

import ast
import dataclasses
import io
import tokenize
from collections.abc import Iterator, MutableSet, Sequence

from absl import logging

#用于后面的 dataclass 的定位，有关键字定位到该类
@dataclasses.dataclass
class Function:
    """一个已解析的 Python 函数。"""

    name: str
    args: str
    body: str
    return_type: str | None = None
    docstring: str | None = None    # 文档字符串

    def __str__(self) -> str:     # 将函数变为字符串
        return_type = f" -> {self.return_type}" if self.return_type else ""

        function = f"def {self.name}({self.args}){return_type}:\n"
        if self.docstring:
            # self.docstring 已经缩进，除了第一行外每行都缩进了。
            # 这里我们假设缩进始终是两个空格。
            new_line = "\n" if self.body else ""
            function += f'  """{self.docstring}"""{new_line}'
        # self.body 已经缩进。
        function += self.body + "\n\n"
        return function

#这个时 python 特有的，只要有新的对象被构造，这个方法就会被触发
    def __setattr__(self, name: str, value: str) -> None:
        # 确保 `body` 中没有前导或尾随换行符。
        if name == "body":
            value = value.strip("\n")
        # 确保 `docstring` 中没有前导或尾随的三引号。
        if name == "docstring" and value is not None:
            if '"""' in value:
                value = value.strip()
                value = value.replace('"""', "")
        #这个方法是所有父类都有，父类的该方法是写入数据
        super().__setattr__(name, value)


@dataclasses.dataclass(frozen=True)
class Program:
    """一个已解析的 Python 程序。"""

    # `preface` 是从代码开头到第一个函数定义之前的所有内容。
    preface: str
    functions: list[Function]

#还原成字符串
    def __str__(self) -> str:
        program = f"{self.preface}\n" if self.preface else ""
        program += "\n".join([str(f) for f in self.functions])  # 调用了 Function 里面的 str 方法
        return program

#返回某个函数的索引——告诉一个函数在程序中的位置
    def find_function_index(self, function_name: str) -> int:
        """返回输入函数名对应的索引。"""
        function_names = [f.name for f in self.functions]
        count = function_names.count(function_name)
        if count == 0:
            raise ValueError(f"function {function_name} does not exist in program:\n{str(self)}")
        if count > 1:
            raise ValueError(f"function {function_name} exists more than once in program:\n{str(self)}")
        index = function_names.index(function_name)
        return index

#这个是 Function 类的对外接口
    def get_function(self, function_name: str) -> Function:
        index = self.find_function_index(function_name)
        return self.functions[index]

# 遍历字符串工具——进而实现 Function 和 perface 的对外接口
class ProgramVisitor(ast.NodeVisitor):
    """解析代码以收集生成 `Program` 所需的全部信息。

    注意，我们不存储函数装饰器。
    """

    def __init__(self, sourcecode: str):
        self._codelines: list[str] = sourcecode.splitlines()

        self._preface: str = ""     # 存储第一个顶层函数上的东西--前言
        self._functions: list[Function] = []    # 存储数据的地方--存储顶层函数
        self._current_function: str | None = None

# ast 的经典操作，主要是为了得到一个程序中的函数列表，其次，也将函数全部封装成定义好的 function 类
    def visit_FunctionDef(
        self,  # pylint: disable=invalid-name
        node: ast.FunctionDef,
    ) -> None:
        """收集正在解析的函数的全部信息。"""
        if node.col_offset == 0:  # 我们只关注顶层函数。
            self._current_function = node.name
            if not self._functions: # 此时 prefase 为空
                self._preface = "\n".join(self._codelines[: node.lineno - 1])
            function_end_line = node.end_lineno # 结束行号
            body_start_line = node.body[0].lineno - 1
            # 提取文档字符串。
            docstring = None
            # isinstance用来判断该对象属于什么类型
            if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                # 这里就有可能存在双重"""
                docstring = f'  """{ast.literal_eval(ast.unparse(node.body[0]))}"""'
                if len(node.body) > 1:
                    body_start_line = node.body[1].lineno - 1
                else:
                    body_start_line = function_end_line

            self._functions.append(
                Function(
                    name=node.name,
                    args=ast.unparse(node.args),
                    return_type=ast.unparse(node.returns) if node.returns else None,
                    docstring=docstring,
                    body="\n".join(self._codelines[body_start_line:function_end_line]),
                )
            )
        # 继续遍历子节点
        self.generic_visit(node)        # 这个其实没有什么鸟用

#将 perface 和 Function 列表一并作为对外的接口
    def return_program(self) -> Program:
        return Program(preface=self._preface, functions=self._functions)

# 代码字符串转 ast 抽象树的入口
def text_to_program(text: str) -> Program:
    """使用 Python AST 解析输入文本，返回 Program 对象。"""
    try:
        # 我们假设程序由一些前言（例如导入、类、赋值等）组成，随后是一系列函数。
        tree = ast.parse(text)
        visitor = ProgramVisitor(text)
        visitor.visit(tree) # 这一步完成了 Function 类的封装【其实利用了 visit 这种回调函数】
        return visitor.return_program()
    except Exception as e:
        logging.warning("Failed parsing %s", text)
        raise e

# 特殊！！！只用于得到只有一个函数的程序，且返回的内部的唯一的函数
# 用于处理进化函数
def text_to_function(text: str) -> Function:
    """使用 Python AST 解析输入文本，返回 Function 对象。"""
    program = text_to_program(text)
    if len(program.functions) != 1:
        raise ValueError(f"Only one function expected, got {len(program.functions)}:\n{program.functions}")
    return program.functions[0]

# 把代码串拆成单词，为了更精细的处理
def _tokenize(code: str) -> Iterator[tokenize.TokenInfo]:
    """将 `code` 转换为 Python 词法单元。"""
    code_bytes = code.encode()
    code_io = io.BytesIO(code_bytes)
    return tokenize.tokenize(code_io.readline)

# 解开
def _untokenize(tokens: Sequence[tokenize.TokenInfo]) -> str:
    """将 Python 词法单元列表转换回代码。"""
    return tokenize.untokenize(tokens).decode("utf-8")  # 直接返回，它本来就是 str

#利用生成器 yield ，返回可以重命名的函数名
def _yield_token_and_is_call(code: str) -> Iterator[tuple[tokenize.TokenInfo, bool]]:
    """逐个产出词法单元，并带有一个布尔值指示其是否为函数调用。"""
    try:
        tokens = _tokenize(code)
        prev_token = None
        is_attribute_access = False # 是否是属性访问，false表示可以重命名，true表示不可以重命名
        for token in tokens:
            if (
                prev_token  # 如果上一个词法单元存在
                and prev_token.type == tokenize.NAME  # 且它是一个 Python 标识符
                and token.type == tokenize.OP  # 且当前词法单元是一个分隔符
                and token.string == "("
            ):  # 并且具体是 '('。
                yield prev_token, not is_attribute_access
                is_attribute_access = False # 对应的是外部的 iscall ，false 表示不可以重命名，true 则可以
            else:
                if prev_token:
                    is_attribute_access = prev_token.type == tokenize.OP and prev_token.string == "."
                    yield prev_token, False
            prev_token = token
        if prev_token:
            yield prev_token, False
    except Exception as e:
        logging.warning("Failed parsing %s", code)
        raise e

# 修改好需要修改的，并返回完整的修改后的字符串
def rename_function_calls(code: str, source_name: str, target_name: str) -> str:
    """将函数调用从 `source_name` 重命名为 `target_name`。"""
    if source_name not in code:
        return code
    modified_tokens = []    # 修改过的 tokens
    for token, is_call in _yield_token_and_is_call(code):
        if is_call and token.string == source_name:
            # 替换函数名所在的词法单元
            modified_token = tokenize.TokenInfo(
                type=token.type,
                string=target_name,
                start=token.start,
                end=token.end,
                line=token.line,
            )
            modified_tokens.append(modified_token)
        else:
            modified_tokens.append(token)
    return _untokenize(modified_tokens)

# 返回 code 中调用的所有函数的集合
def get_functions_called(code: str) -> MutableSet[str]:
    """返回 `code` 中调用的所有函数名集合。"""
    return set(token.string for token, is_call in _yield_token_and_is_call(code) if is_call)

# 返回被装饰器 @ 装饰的函数名
def yield_decorated(code: str, module: str, name: str) -> Iterator[str]:
    """产出 `code` 中被 `@module.name` 装饰的函数名。"""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                attribute = None
                if isinstance(decorator, ast.Attribute):
                    attribute = decorator
                elif isinstance(decorator, ast.Call):
                    attribute = decorator.func
                if attribute is not None and attribute.value.id == module and attribute.attr == name:
                    yield node.name



# def part(x) -> int :
#     """
#     文档介绍串
#     """
#     for x in [1,3,6]:
#         # x是一个外界变量
#         print(f"{x}+1")
#     return x
