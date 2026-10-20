import ast
from implementation.evaluator import Sandbox
from demo_1.dataset import datasets

class DummySandbox(Sandbox):
    def __init__(self) -> None:
        self.datasets=datasets
        
    def run(self, program: str, function_to_run: str, test_input: str, timeout_seconds: int):
        # 1. 创建执行环境（命名空间）
        namespace = {}
        namespace['DATASET']=self.datasets

        try:
            # 2. 解析、编译并执行整个程序
            parsed_code = ast.parse(program)
            compiled_code = compile(parsed_code, filename="<ast>", mode="exec")
            exec(compiled_code, namespace)

            # 3. 从命名空间中获取目标函数并执行
            if function_to_run in namespace:
                result = namespace[function_to_run](test_input)
                return result, True
            else:
                # 目标函数未找到
                return None, False
        except Exception:
            # 任何执行错误都视为失败
            return None, False
