from pprint import pp

from langchain_community.document_loaders.web_base import WebBaseLoader

url = "https://reference.langchain.com/python/langchain-community/document_loaders/web_base/WebBaseLoader"

web_loader = WebBaseLoader(web_path=url)
# web_loader = WebBaseLoader(web_paths=[url1, url2, url3])

documents = web_loader.load()

pp(len(documents))  # <--- one doc / url

pp(documents)

# [Document(metadata={'source': 'https://reference.langchain.com/python/langchain-community/document_loaders/web_base/WebBaseLoader', 'title': 'WebBaseLoader | langchain_community | LangChain Reference', 'description': 'Python API reference for document_loaders.web_base.WebBaseLoader in langchain_community. Part of the LangChain ecosystem.', 'language': 'en'}, page_content='WebBaseLoader | langchain_community | LangChain ReferenceLangChain Reference home pageSearch...⌘KAsk AIGitHubMain DocsDeep AgentsLangChainLangGraphIntegrationsLangSmithOverviewAmazon NovaAnthropicAstraDBAWSAzure (Microsoft)CerebrasChromaCohereCommunityOverviewChat ModelsLLMsEmbeddingsVector StoresDocument LoadersTools & AgentsRetrieversCachesChat HistoryGraphsUtilitiesDb2DeepSeekElasticsearchExaFireworksGoogle (Community)Google GenAI (Gemini)Google Vertex AIGroqHuggingFaceIBMLiteLLMMilvusMistral AINeo4JNomicNvidia AI EndpointsOllamaOpenAIOpenRouterParallelPerplexityPineconePostgresQdrantRedisSema4SnowflakeSQLServerTavilyTogetherUnstructuredUpstageWeaviatexAI⌘ILangChain AssistantNewHistoryAsk a question to get startedEnter to send•Shift+Enter new lineMenuNavigationProjectsAmazon NovaAnthropicAstraDBAWSAzure (Microsoft)CerebrasChromaCohereCommunityOverviewChat ModelsLLMsEmbeddingsVector StoresDocument LoadersTools & AgentsRetrieversCachesChat HistoryGraphsUtilitiesDb2DeepSeekElasticsearchExaFireworksGoogle (Community)Google GenAI (Gemini)Google Vertex AIGroqHuggingFaceIBMLiteLLMMilvusMistral AINeo4JNomicNvidia AI EndpointsOllamaOpenAIOpenRouterParallelPerplexityPineconePostgresQdrantRedisSema4SnowflakeSQLServerTavilyTogetherUnstructuredUpstageWeaviatexAILanguagePythonJavaScriptThemeLightDarkPythonlangchain-communitydocument_loadersweb_baseWebBaseLoaderClassv0.4.2 (latest)●Since v0.3WebBaseLoaderCopyWebBaseLoader(\n  self,\n  web_path: Union[str, Sequence[str]] = \'\'BasesBaseLoaderUsed in DocsGoogle cloud Vertex AI reranker integrationConstructorsAttributesMethodsInherited fromBaseLoader(langchain_core)MethodsMloadMload_and_splitView source on GitHubVersion History,\nheader_template: Optional[dict] = None,\nverify_ssl: bool = True,\nproxies: Optional[dict] = None,\ncontinue_on_failure: bool = False,\nautoset_encoding: bool = True,\nencoding: Optional[str] = None,\nweb_paths: Sequence[str] = (),\nrequests_per_second: int
