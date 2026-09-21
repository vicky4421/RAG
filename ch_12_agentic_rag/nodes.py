from enum import StrEnum


class Node(StrEnum):
    RETRIEVE = "retrieve"
    GENERATE_RESPONSE = "generate_response"
    DECIDE_RETRIEVAL = "decide_retrieval"


class RouteVerdict(StrEnum):
    NEED_RETRIEVAL = "need_retrieval"
    GENERATE_DIRECT = "direct_generate"
