from implementation.sampler import LLM
import os
from openai import OpenAI


#从环境变量中获得密钥
key=os.environ.get("DEEPSEEK_KEY")

class DeepSeekLLM(LLM):
    """ 接入 deepseek 的 LLM """
    def __init__(self,samples_per_prompt:int =4,
                 model:str="deepseek-v4-flash",
                 temperature:float=1.2,#较高，生成更随机，探索模式
                 max_tokens:int=2048,#回答的最大限制
                 api_key=key):
        # super 是可以调用父类一个方法的中介
        super().__init__(samples_per_prompt)

        #获得，密钥
        self.api_key=api_key
        if not self.api_key:
            raise ValueError("DeepSeek API key must be provided or set in environment variable DEEPSEEK_API_KEY")

        self.client=OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
            )
        self.model=model
        # temperature 控制输出结果的随机性，数值越低，AI 越保守，数值越高，越天马行空
        self.temperature=temperature
        self.max_tokens=max_tokens

    def _draw_sample(self, prompt: str) -> str:
        """调用 DeepSeek API 生成补全结果"""

        message={
            "role":""
        }

        try:
            response=self.client.chat.completions.create(
                model=self.model,
                # 系统提示，自定义
                # system:人为设定，user：当前用户的真实提问，而内容是一个 prompt
                messages=[
                    {
                        "role":"system",
                        "content":
                        "You are a Python code completion engine for an evolutionary search system.\n"
                        "The user message is source code ending inside an incomplete function body.\n"
                        "Continue exactly from the end of that code, and output ONLY the missing "
                        "indented Python function body.\n"
                        "Rules:\n"
                        "- Use 2-space indentation, matching the template.\n"
                        "- Do not reprint anything already present: no imports, no function signature, "
                        "no docstring, no earlier _v0/_v1 versions.\n"
                        "- Do not output markdown, code fences, explanations, or any text before/after "
                        "the code.\n"
                        "- Start directly with the indented statements and stop when the function body "
                        "is complete."
                    },
                     {"role":"user","content":prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False,    # 不流式，直接拿完整结果
                extra_body={"thinking": {"type": "disabled"}}
                #这里不开思考模式
            )
            # 从响应中提取生成的文本
            # return (response.choices[0].message.content or "").strip()
            # print(f"  [第一处测试]: {response.choices[0].finish_reason}")
            # print(f"  [API model]: {response.model}")
            content = response.choices[0].message.content or "" #获得字符串代码块
            return content.strip()      #移除首尾空格


        except Exception as e:
            print(f"DeepSeek API error: {e}")
            raise e
