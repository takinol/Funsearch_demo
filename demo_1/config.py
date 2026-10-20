import implementation.config as cf

ProgramsDatabaseConfig=cf.ProgramsDatabaseConfig(
    functions_per_prompt=2,
    num_islands=10,
    reset_period=3600,
    cluster_sampling_temperature_init=3,#提高簇温度，进而更容易实现交叉
    cluster_sampling_temperature_period=80# 200个程序一次温度线性变化
)

get_config=cf.Config(
    programs_database=ProgramsDatabaseConfig,
    num_evaluators=140,
    num_samplers=15,
    samples_per_prompt=4
)


# """解决某问题。。。"""
# def priority_v0():
#     #对应的程序Body


# def priority_v1():
#     #由上一个程序进化而来
#     #对应程序Body

# def priority_v2():
#     #由上一个程序进化而来
#     #由LLM补充