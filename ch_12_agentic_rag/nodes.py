from enum import StrEnum


class Node(StrEnum):
    RETRIEVE = "retrieve"
    GENERATE_RESPONSE = "generate_response"
    DECIDE_RETRIEVAL = "decide_retrieval"
    AGENT = "agent"
    TOOL = "tool"
    TOOL_OUTPUT = "tool_output"


class RouteVerdict(StrEnum):
    NEED_RETRIEVAL = "need_retrieval"
    GENERATE_DIRECT = "direct_generate"
    TOOLS_REQUIRED = "tools_required"
    TOOLS_COMPLETE = "tools_complete"
