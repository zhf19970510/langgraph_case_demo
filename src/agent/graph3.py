import asyncio
import json
from typing import Dict, Any, List

from langchain_core.messages import ToolMessage, AIMessage, ToolCall
from langchain_mcp_adapters.client import MultiServerMCPClient
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
    graph = builder.compile()
    return graph


agent = asyncio.run(create_graph())