import asyncio
import json
from typing import Dict, Any, List

from langchain_core.messages import ToolMessage, AIMessage, ToolCall
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.env_utils import ZHIPU_API_KEY
from agent.my_llm import llm

# 外网上公开 MCP 服务端的连接配置
zhipuai_mcp_server_config = {
    'url': 'https://open.bigmodel.cn/api/mcp/web_search/sse?Authorization='+ZHIPU_API_KEY,
    'transport': 'sse',
}

my12306_mcp_server_config = {
    'url': 'https://mcp.api-inference.modelscope.net/26401a0eff2846/mcp',
    'transport': 'streamable_http',
}

chart_mcp_server_config = {
    'url': 'https://mcp.api-inference.modelscope.net/4e05cc1f5e2044/sse',
    'transport': 'sse',
}

# MCP的客户端
mcp_client = MultiServerMCPClient(
    {
        'chart_mcp': chart_mcp_server_config,
        'my12306_mcp': my12306_mcp_server_config,
        'zhipuai_mcp': zhipuai_mcp_server_config,
    }
)



class State(MessagesState):
    pass



async def create_graph():
    tools = await mcp_client.get_tools()  # 30个以上的工具，全部来自MCP服务端

    builder = StateGraph(State)

    llm_with_tools = llm.bind_tools(tools)

    async def chatbot(state: State):
        return {'messages': [ await llm_with_tools.ainvoke(state["messages"])]}


    builder.add_node('chatbot', chatbot)

    # tool_node = BasicToolsNode(tools)
    tool_node = ToolNode(tools=tools)
    builder.add_node('tools', tool_node)

    builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    builder.add_edge('tools', 'chatbot')
    builder.add_edge(START, 'chatbot')

    # 检查点让状态图可以持久化其状态
    # 这是整个状态图的完整内存
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory, interrupt_before=['tools'])
    return graph


# agent = asyncio.run(create_graph())

async def run_graph():
    graph = await create_graph()
    # 配置参数，包含乘客ID和线程ID
    config = {
        "configurable": {
            # 检查点由session_id访问
            "thread_id": 'zs12311',
        }
    }

    def get_answer(tool_message, user_answer):
        """让人工介入，并且给一个问题的答案"""

        tool_name = tool_message.tool_calls[0]['name']
        answer = (
            f"人工强制终止了工具：{tool_name}的执行，拒绝的理由是：{user_answer}"
        )

        # 创建一个消息
        new_message = [
            ToolMessage(content=answer, tool_call_id=tool_message.tool_calls[0]['id']),
            AIMessage(content=answer)
        ]

        # 把新人造的消息，添加到工作流的state中
        graph.update_state(  # 手动修改state
            config=config,
            values={'messages': new_message}
        )

    def print_message(event, result):
        """格式化输出消息"""
        messages = event.get('messages')
        if messages:
            if isinstance(messages, list):
                message = messages[-1]  # 如果消息是列表，则取最后一个
            if message.__class__.__name__ == 'AIMessage':
                if message.content:
                    # print(result)
                    result = message.content  # 需要在展示的消息
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > 1500:
                msg_repr = msg_repr[:1500] + " ... （已截断）"  # 超过最大长度则截断
            print(msg_repr)  # 输出消息的表示形式
        return result

    async def execute_graph(user_input: str) -> str:
        """ 执行工作流的函数"""
        result = '' # AI助手的最后一条消息
        # stream_mode='values' 会把中间的一些工具和AIMessage都给显示出来
        if user_input.strip().lower() != 'y':  # 正常的用户提问
            current_state = graph.get_state(config)
            if current_state.next:  # 如果有下一步，则当前工作流处在中断中
                tools_script_message = current_state.values['messages'][-1]  # 状态中存储的最后一个message
                # 通过提供关于请求的更改/改变主意的指示来满足工具调用
                get_answer(tools_script_message, user_input)
                message = graph.get_state(config).values['messages'][-1]
                result = message.content

                return result
            else:
                # 不是可等待对象（Awaitable），因此不能直接用于 await 表达式, 必须通过 async for 迭代
                async for chunk in graph.astream({'messages': ('user', user_input)}, config, stream_mode='values'):
                    result = print_message(chunk, result)

        else:  # 用户输入了Y 想继续工具的调用
            async for chunk in graph.astream(None, config, stream_mode='values'):
                result = print_message(chunk, result)

        current_state = graph.get_state(config)
        if current_state.next:  # 出现了工作流的中断
            ai_message = current_state.values['messages'][-1]
            tool_name = ai_message.tool_calls[0]['name']
            # ai_message.tool_calls[0]['args']
            result = f"AI助手马上根据你要求，执行{tool_name}工具。您是否批准继续执行？输入'y'继续；否则，请说明您理由。\n"

        return result

    # 执行工作流
    while True:
        user_input = input('用户：')
        res = await execute_graph(user_input)
        print('AI: ', res)


if __name__ == '__main__':
    asyncio.run(run_graph())