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
    DECIDE_DECOMPOSITION = "decide_decomposition"
    DECOMPOSE = "decompose"
    BUILD_PROMPT = "build_prompt"


class RouteVerdict(StrEnum):
    RETRIEVAL_NEEDED = "retrieval_needed"
    RETRIEVAL_NOT_NEEDED = "retrieval_not_needed"
    GENERATE_DIRECT = "direct_generate"
    TOOLS_REQUIRED = "tools_required"
    TOOLS_COMPLETE = "tools_complete"
    TOOL_CALL_LIMIT_REACHED = "tool_call_limit_reached"
    TOOL_CALL_WITHIN_LIMIT = "tool_call_within_limit"
    DOCS_RELEVANT = "docs_relevant"
    DOCS_NOT_RELEVANT = "docs_not_relevant"
    REWRITE_EXHAUSTED = "rewrite_exhasted"
    REWRITE_NEEDED = "rewrite_needed"
    DECOMPOSITION_NEEDED = "decomposition_needed"
    DECOMPOSITION_NOT_NEEDED = "decomposition_not_needed"
