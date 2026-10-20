

from implementation import programs_database,code_manipulation,config,evaluator,sampler
from demo_1 import config,dataset,llm_deepseek,sandbox,specification,visual

from demo_1.config import get_config
from demo_1.dataset import datasets
from demo_1.specification import specification as SPEC
from demo_1.sandbox import DummySandbox
from demo_1.llm_deepseek import DeepSeekLLM

import copy


cfg=get_config
template=code_manipulation.text_to_program(SPEC)
objective_function=template.get_function("priority")
#构建数据库
db=programs_database.ProgramsDatabase(
    config=cfg.programs_database,
    template=template,
    function_to_evolve="priority")
#数据库初始化
heuristic_priority={
    "Best Fit":"  return -(bins - item)\n",
    "Fast Fit":"  return np.zeros_like(bins)\n",
}

#实现了在一个岛里面，同时有着两种不同的装箱算法
for name,body in heuristic_priority.items():
    #组建好不同装箱方法的函数
    priority_function=copy.deepcopy(objective_function)
    priority_function.body=body

    #组装好完整的程序
    prog=copy.deepcopy(template)
    idx=prog.find_function_index("priority")
    prog.functions[idx]=copy.deepcopy(priority_function)

    sandbox_=sandbox.DummySandbox()
    score_test={}

    for dataset_name in ['OR3','Weibull 5k']:
        score_,ok=sandbox_.run(
            program=str(prog),
            function_to_run="evaluate",
            test_input=dataset_name,
            timeout_seconds=30
        )
        if not ok or score_ is None:
            raise RuntimeError(f'{name} 在 {dataset_name} 上评估失败')
        score_test[dataset_name]=score_

    print(f"{name}\t:{score_test}")

    db.register_program(
        program=copy.deepcopy(priority_function),
        island_id=None,
        scores_per_test=score_test)

#初始化评估器
evaluators=[]
for _ in range(cfg.num_evaluators):
    evaluators.append(evaluator.Evaluator(
        database=db,
        template=template,
        function_to_evolve="priority",
        function_to_run="evaluate",
        inputs=['OR3','Weibull 5k'],# 这里只取一个数据集
        timeout_seconds=30,#没用
        sandbox=DummySandbox()
    ))

s=sampler.Sampler(
    database=db,
    evaluators=evaluators,
    samples_per_prompt=cfg.samples_per_prompt,
    llm=DeepSeekLLM()
)
s.sample(max_iterations=60)


