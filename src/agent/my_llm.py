from langchain_openai import ChatOpenAI

from agent.env_utils import LOCAL_BASE_URL, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, OPENAI_API_KEY, OPENAI_BASE_URL

# openai的大模型
llm = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0.8,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

# deepseek大模型
# deepseek大模型
# llm = ChatOpenAI(
#     model='deepseek-chat',
#     temperature=0.8,
#     api_key=OPENAI_API_KEY,
#     base_url=OPENAI_BASE_URL
# )


#  claude 的大模型调用
# llm = ChatOpenAI(
#     model='claude-3-7-sonnet-20250219',
#     temperature=0.8,
#     api_key=OPENAI_API_KEY,
#     base_url=OPENAI_BASE_URL,
# )

# 官方的deepseek
# llm = ChatOpenAI(
#     model='deepseek-reasoner',
#     # model='deepseek-chat',
#     temperature=0.8,
#     api_key=DEEPSEEK_API_KEY,
#     base_url=DEEPSEEK_BASE_URL,
#     # model_kwargs={ "response_format": { "type": "json_object" } },
# )

# 本地vllm 私有化部署的大模型: 采用--tool-call-parser == hermes
# 流式输出的时候，会有错误
# llm = ChatOpenAI(
#     model='qwen3-8b',
#     temperature=0.8,
#     api_key='xx',
#     base_url=LOCAL_BASE_URL,
#     extra_body={'chat_template_kwargs': {'enable_thinking': True}},
# )

# 本地sglang 本地私有化部署的大模型： 采用--tool-call-parser == qwen25
# llm = ChatOpenAI(
#     model='qwen3-8b',
#     temperature=0.8,
#     api_key='xx',
#     base_url='http://i-2.gpushare.com:42124/v1',
#     extra_body={'chat_template_kwargs': {'enable_thinking': True}},
# )

# llm = ChatOpenAI(
#     model='ds-qwen3-8b',
#     temperature=0.5,
#     api_key='',
#     base_url=LOCAL_BASE_URL,
#     # extra_body={'chat_template_kwargs': {'enable_thinking': True}},
# )

# multiModal_llm = ChatOpenAI(  # 多模态大模型
#     model='qwen-omni-3b',
#     api_key='xx',
#     base_url=LOCAL_BASE_URL,
# )