import asyncio
import json
from typing import Dict, Any, List
import gradio as gr
from gradio import ChatMessage
from langchain_core.messages import ToolMessage, AIMessage, ToolCall, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.types import interrupt, Command

from agent.env_utils import ZHIPU_API_KEY
from agent.my_llm import llm

# 外网上公开 MCP 服务端的连接配置
zhipuai_mcp_server_config = {
    'url': 'https://open.bigmodel.cn/api/mcp/web_search/sse?Authorization=' + ZHIPU_API_KEY,
    'transport': 'sse',
}

my12306_mcp_server_config = {
    'url': 'https://mcp.api-inference.modelscope.net/f938a50626714f/sse',
    'transport': 'sse',
}

chart_mcp_server_config = {
    'url': 'https://mcp.api-inference.modelscope.net/c76cefe1dbae4f/sse',
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


class BasicToolsNode:
    """
    异步工具节点，用于并发执行AIMessage中请求的工具调用

    功能：
    1. 接收工具列表并建立名称索引
    2. 并发执行消息中的工具调用请求
    3. 自动处理同步/异步工具适配
    """

    def __init__(self, tools: list):
        """初始化工具节点
        Args:
            tools: 工具列表，每个工具需包含name属性
        """
        self.tools_by_name = {tool.name: tool for tool in tools}  # 所有工具名字的字典

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, List[ToolMessage]]:
        """异步调用入口
        Args:
            state: 输入字典，需包含"messages"字段
        Returns:
            包含ToolMessage列表的字典
        Raises:
            ValueError: 当输入无效时抛出
        """
        # 1. 输入验证
        if not (messages := state.get("messages")):
            raise ValueError("输入数据中未找到消息内容")  # 改进后的中文错误提示
        message: AIMessage = messages[-1]  # 取最新消息: AIMessage

        tool_name = message.tool_calls[0]["name"] if message.tool_calls else None
        if tool_name == 'webSearchStd' or tool_name == 'webSearchSogou':
            response = interrupt(
                f"AI大模型尝试调用工具 `{tool_name}`，\n"
                "请审核并选择：批准（y）或直接给我工具执行的答案。"
            )
            # response(字典): 由人工输入的：批准(y),工具执行的答案或者拒绝执行工具的理由
            # 根据人工响应类型处理
            if response["answer"] == "y":
                pass  # 直接使用原参数继续执行
            else:
                return {"messages": [ToolMessage(
                    content=f"人工终止了该工具的调用，给出的理由或者答案是:{response['answer']}",
                    name=tool_name,
                    tool_call_id=message.tool_calls[0]['id'],
                )]}

        # 2. 并发执行工具调用
        outputs = await self._execute_tool_calls(message.tool_calls)
        return {"messages": outputs}

    async def _execute_tool_calls(self, tool_calls: list[Dict]) -> List[ToolMessage]:
        """执行实际工具调用
        Args:
            tool_calls: 工具调用请求列表
        Returns:
            ToolMessage结果列表
        """

        async def _invoke_tool(tool_call: Dict) -> ToolMessage:
            """执行单个工具调用
            Args:
                tool_call: 工具调用请求字典，需包含name/args/id字段
            Returns:
                封装的ToolMessage
            Raises:
                KeyError: 工具未注册时抛出
                RuntimeError: 工具调用失败时抛出
            """
            try:
                # 3. 异步调用工具
                tool = self.tools_by_name.get(tool_call["name"])  # 验证 工具是否在之前的 工具集合中
                if not tool:
                    raise KeyError(f"未注册的工具: {tool_call['name']}")

                if hasattr(tool, 'ainvoke'):  # 优先使用异步方法
                    tool_result = await tool.ainvoke(tool_call["args"])
                else:  # 同步工具通过线程池转异步
                    loop = asyncio.get_running_loop()
                    tool_result = await loop.run_in_executor(
                        None,  # 使用默认线程池
                        tool.invoke,  # 同步调用方法
                        tool_call["args"]  # 参数
                    )

                # 4. 构造ToolMessage
                return ToolMessage(
                    content=json.dumps(tool_result, ensure_ascii=False),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            except Exception as e:
                print(e)
                raise RuntimeError(f"工具调用失败: {tool_call['name']}") from e

        try:
            # 5. 并发执行所有工具调用
            # asyncio.gather() 是 Python 异步编程中用于并发调度多个协程的核心函数，其核心行为包括：
            # 并发执行：所有传入的协程会被同时调度到事件循环中，通过非阻塞 I/O 实现并行处理。
            # 结果收集：按输入顺序返回所有协程的结果（或异常），与任务完成顺序无关。
            # 异常处理：默认情况下，任一的任务失败会立即取消其他任务并抛出异常；若设置 return_exceptions=True，则异常会作为结果返回。
            #
            return await asyncio.gather(*[_invoke_tool(tool_call) for tool_call in tool_calls])
        except Exception as e:
            print(e)
            raise RuntimeError("并发执行工具时发生错误") from e


class State(MessagesState):
    pass


def route_tools_func(state: State):
    """
    动态路由函数，如果从大模型输出后的AIMessage，中包含有工具调用的请求(指令)， 就进入到tools节点， 否则则结束
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


async def create_graph():
    tools = await mcp_client.get_tools()  # 30个以上的工具，全部来自MCP服务端

    builder = StateGraph(State)

    llm_with_tools = llm.bind_tools(tools)

    async def chatbot(state: State):
        return {'messages': [await llm_with_tools.ainvoke(state["messages"])]}

    builder.add_node('chatbot', chatbot)

    tool_node = BasicToolsNode(tools)
    builder.add_node('tools', tool_node)

    builder.add_conditional_edges(
        "chatbot",
        route_tools_func,
        {"tools": "tools", END: END}
    )
    builder.add_edge('tools', 'chatbot')
    builder.add_edge(START, 'chatbot')
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    return graph


graph = asyncio.run(create_graph())
 # 配置参数，包含乘客ID和线程ID
config = {
    "configurable": {
        # 检查点由session_id访问
        "thread_id": 'zs12311',
    }
}

def add_message(chat_history, user_message):
    if user_message:
        chat_history.append({"role": "user", "content": user_message})
    return chat_history, gr.Textbox(value=None, interactive=False)


async def submit_messages(chat_history: List[Dict]):
    """流式处理消息的核心函数"""
    user_input = chat_history[-1]['content']
    current_state = graph.get_state(config)
    full_response = ""  # 累积完整响应
    tool_calls = []  # 记录工具调用

    # 处理中断恢复或正常消息
    inputs = Command(resume={'answer': user_input}) if current_state.next else {
        'messages': [HumanMessage(content=user_input)]}

    async for chunk in graph.astream(
            inputs,
            config,
            stream_mode=["messages", "updates"],  # 同时监听消息和状态更新
    ):
        if 'messages' in chunk:
            for message in chunk[1]:
                # 处理AI消息流式输出
                if isinstance(message, AIMessage) and message.content:
                    full_response += message.content
                    # 更新最后一条消息而非追加
                    if chat_history and isinstance(chat_history[-1], ChatMessage) and 'title' not in chat_history[-1].metadata:
                        chat_history[-1].content = full_response
                    else:
                        chat_history.append(ChatMessage(role="assistant", content=message.content))
                    yield chat_history

                # 处理工具调用消息
                elif isinstance(message, ToolMessage):
                    tool_msg = f"🔧 调用工具: {message.name}\n{message.content}"
                    chat_history.append(ChatMessage(role="assistant", content=tool_msg,
                                        metadata={"title": f"🛠️ Used tool {message.name}"}))
                    yield chat_history

    # 检查新中断
    current_state = graph.get_state(config)
    if current_state.next:
        interrupt_msg = current_state.interrupts[0].value
        # chat_history.append({'role': 'assistant', 'content': interrupt_msg})
        chat_history.append(ChatMessage(role="assistant", content=interrupt_msg))
        yield chat_history


# 创建Gradio界面
with gr.Blocks(
        title='我的智能小秘书',
        theme=gr.themes.Soft(),
        css=".system {color: #666; font-style: italic;}"  # 自定义系统消息样式
) as demo:
    # 聊天历史记录组件
    chatbot = gr.Chatbot(
        type="messages",
        height=500,
        render_markdown=True,  # 支持Markdown格式
        line_breaks=False  # 禁用自动换行符
    )

    # 输入组件
    chat_input = gr.Textbox(
        placeholder="请输入您的消息...",
        label="用户输入",
        max_lines=5,
        container=False
    )

    # 控制按钮
    with gr.Row():
        submit_btn = gr.Button("发送", variant="primary")
        clear_btn = gr.Button("清空对话")

    # 消息提交处理链
    msg_handler = chat_input.submit(
        fn=add_message,
        inputs=[chatbot, chat_input],
        outputs=[chatbot, chat_input],
        queue=False
    ).then(
        fn=submit_messages,
        inputs=chatbot,
        outputs=chatbot,
        api_name="chat_stream"  # API端点名称
    )

    # 按钮点击处理链
    btn_handler = submit_btn.click(
        fn=add_message,
        inputs=[chatbot, chat_input],
        outputs=[chatbot, chat_input],
        queue=False
    ).then(
        fn=submit_messages,
        inputs=chatbot,
        outputs=chatbot
    )

    # 清空对话
    clear_btn.click(
        fn=lambda: [],
        inputs=None,
        outputs=chatbot,
        queue=False
    )

    # 重置输入框状态
    msg_handler.then(
        lambda: gr.Textbox(interactive=True),
        None,
        [chat_input]
    )
    btn_handler.then(
        lambda: gr.Textbox(interactive=True),
        None,
        [chat_input]
    )

if __name__ == '__main__':
    demo.launch()