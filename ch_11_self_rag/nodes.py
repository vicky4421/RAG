from enum import StrEnum


# Define nodes
class Node(StrEnum):
    DECIDE_RETRIEVAL = "decide_retrieval"
    GENERATE_DIRECT = "generate_direct"
    RETRIEVE = "retrieve"
    IS_RELEVANT = "is_relevant_node"


# Define your Routing Decisions as typed constants
class RouteVerdict(StrEnum):
    DIRECT = "generate_direct"
    RETRIEVE = "retrieve"
