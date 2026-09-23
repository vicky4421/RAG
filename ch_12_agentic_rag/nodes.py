from enum import StrEnum


class Node(StrEnum):
    RETRIEVE = "retrieve"
    GENERATE_RESPONSE = "generate_response"
    DECIDE_RETRIEVAL = "decide_retrieval"
    AGENT = "agent"
    TOOL = "tool"
    TOOL_OUTPUT = "tool_output"
    INCREMENT_TOOL_CALL = "increment_tool_call"
    EVALUATE_DOCS = "evaluate_docs"
    REWRITE_QUERY = "rewrite_query"


class RouteVerdict(StrEnum):
    NEED_RETRIEVAL = "need_retrieval"
    GENERATE_DIRECT = "direct_generate"
    TOOLS_REQUIRED = "tools_required"
    TOOLS_COMPLETE = "tools_complete"
    TOOL_CALL_LIMIT_REACHED = "tool_call_limit_reached"
    DOCS_ARE_RELEVANT = "docs_are_relevant"
    DOCS_ARE_NOT_RELEVANT = "docs_are_not_relevant"
    DO_NOT_REWRITE = "don't_rewrite"
    NEED_TO_REWRITE = "need_to_rewrite"
