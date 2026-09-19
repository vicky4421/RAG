from enum import StrEnum


# Define nodes
class Node(StrEnum):
    DECIDE_RETRIEVAL = "decide_retrieval"
    GENERATE_DIRECT = "generate_direct"
    RETRIEVE = "retrieve"
    IS_RELEVANT = "is_relevant_node"
    GENERATE_FROM_CONTEXT = "generate_from_context"
    NO_RELEVANT_DOCS = "no_relevant_docs"
    IS_SUPPORTED = "is_supported"
    ACCEPT_ANSWER = "accept_answer"
    REVISE_ANSWER = "revise_answer"
    IS_USEFUL = "is_useful"


# Define your Routing Decisions as typed constants
class RouteVerdict(StrEnum):
    GENERATE_DIRECT = "generate_direct"
    RETRIEVE = "retrieve"
    GENERATE_FROM_CONTEXT = "generate_from_context"
    NO_RELEVANT_DOCS = "no_relevant_docs"
    ACCEPT_ANSWER = "accept_answer"
    REVISE_ANSWER = "revise_answer"
    NO_ANSWER_FOUND = "no_answer_found"
    END = "END"
