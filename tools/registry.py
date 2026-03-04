"""
Tool Registry - Lightweight tool discovery and invocation mechanism replacing MCP.

All tools are statically registered here. fc_inferencer retrieves schemas and
executor functions via the registry, without MCP protocol for tool discovery
or remote calls.
"""
import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("ToolRegistry")


class ToolRegistry:
    """Lightweight tool registry managing tool schemas and their executor functions."""

    def __init__(self):
        # tool_name -> async callable
        self._executors: Dict[str, Callable] = {}
        # Tool schemas in OpenAI function calling format
        self._schemas: List[Dict[str, Any]] = []

    def register(self, schema: Dict[str, Any], executor: Callable):
        """Register a tool.

        Args:
            schema: Tool schema in OpenAI function calling format
            executor: Corresponding async executor function
        """
        name = schema["function"]["name"]
        if name in self._executors:
            logger.warning(f"Tool '{name}' already registered, skipping")
            return
        self._executors[name] = executor
        self._schemas.append(schema)
        logger.info(f"Registered tool: {name}")

    @property
    def schemas(self) -> List[Dict[str, Any]]:
        return self._schemas

    def get_executor(self, tool_name: str) -> Optional[Callable]:
        return self._executors.get(tool_name)

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._executors

    @property
    def tool_names(self) -> List[str]:
        return list(self._executors.keys())

    async def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a tool and return the string result.

        Args:
            tool_name: Tool name
            args: Tool argument dictionary

        Returns:
            String representation of the tool execution result
        """
        executor = self._executors.get(tool_name)
        if executor is None:
            raise ValueError(f"Tool not found: {tool_name}")

        result = await executor(**args)

        # Convert to string
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)


DEFAULT_TOOLS = ["search", "visit"]

# Available tool name -> builder function mapping
_TOOL_BUILDERS: Dict[str, Callable] = {}


def _register_search(registry: ToolRegistry, config: Dict[str, str]):
    from tools.search import search, configure as search_configure, TOOL_SCHEMA as SEARCH_SCHEMA
    search_configure(serper_api_key=config.get("SERPER_API_KEY", ""))
    registry.register(SEARCH_SCHEMA, search)


def _register_browse(registry: ToolRegistry, config: Dict[str, str]):
    from tools.browse import browse, configure as browse_configure, TOOL_SCHEMA as BROWSE_SCHEMA
    browse_configure(jina_api_key=config.get("JINA_API_KEY", ""))
    registry.register(BROWSE_SCHEMA, browse)


def _register_visit(registry: ToolRegistry, config: Dict[str, str]):
    from tools.web_visitor import visit, configure as visit_configure, TOOL_SCHEMA as VISIT_SCHEMA
    visit_configure(
        jina_api_key=config.get("JINA_API_KEY", ""),
        model_name=config.get("MODEL_NAME", ""),
        base_url=config.get("BASE_URL", ""),
        api_key=config.get("API_KEY", "sk-admin"),
    )
    registry.register(VISIT_SCHEMA, visit)


_TOOL_BUILDERS["search"] = _register_search
_TOOL_BUILDERS["browse"] = _register_browse
_TOOL_BUILDERS["visit"] = _register_visit


def build_default_registry(
    config: Optional[Dict[str, str]] = None,
    tools: Optional[List[str]] = None,
) -> ToolRegistry:
    """Build tool registry, selectively registering tools by the given list.

    Args:
        config: Tool configuration dict containing API keys and other params.
            Supported keys:
            - SERPER_API_KEY: Serper search API key
            - JINA_API_KEY: Jina Reader API key
            - MODEL_NAME: LLM model name for web_visitor
            - BASE_URL: LLM base URL for web_visitor
            - API_KEY: LLM API key for web_visitor
        tools: List of tool names to register. Options: search, browse, visit.
            Default: ["search", "visit"].
    """
    config = config or {}
    tools = tools or DEFAULT_TOOLS
    registry = ToolRegistry()

    for tool_name in tools:
        builder = _TOOL_BUILDERS.get(tool_name)
        if builder is None:
            logger.warning(f"Unknown tool '{tool_name}', skipping. Available: {list(_TOOL_BUILDERS.keys())}")
            continue
        builder(registry, config)

    logger.info(f"Registry built with tools: {registry.tool_names}")
    return registry
