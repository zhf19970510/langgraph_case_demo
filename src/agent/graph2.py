import asyncio
import json
from typing import Any, Dict, List

from langchain_core.messages import ToolMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.env_utils import ZHIPU_API_KEY

# 智能小秘书案例 + MCP工具

# 外网上公开 MCP 服务端的连接配置
zhipuai_mcp_server_config = {
    'url': 'https://open.bigmodel.cn/api/mcp/web_search/sse?Authorization='+ZHIPU_API_KEY,
    'transport': 'sse',
}

# 12306-MCP车票查询工具
my12306_mcp_server_config = {
    'url': 'https://mcp.api-inference.modelscope.net/c0eb3e7389ae4b/mcp',
    'transport': 'streamable_http',
}

# MCP服务器图表
chart_mcp_server_config = {
    'url': 'https://mcp.api-inference.modelscope.net/dc66cd8041024f/sse',
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

# 自定义工具调用：异步、并发、参数验证、异常处理，人工接入等各种功能和要求
class BasicToolsNode:
    """
    异步工具节点：用于并发执行AIMessage中请求的工具调用

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
        self.tools_by_name = {tool.name: tool for tool in tools}  # 所有工具名字的集合

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
            # 异常处理：默认情况下，任一任务失败会立即取消其他任务并抛出异常；若设置 return_exceptions=True，则异常会作为结果返回。
            #
            return await asyncio.gather( *[_invoke_tool(tool_call) for tool_call in tool_calls])
        except Exception as e:
            print(e)
            raise RuntimeError("并发执行工具时发生错误") from e