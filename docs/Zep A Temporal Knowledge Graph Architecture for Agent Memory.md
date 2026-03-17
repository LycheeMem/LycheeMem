Sections:
Abstract
1 Introduction
2 Knowledge Graph Construction
    2.1 Episodes
    2.2 Semantic Entities and Facts
        2.2.1 Entities
        2.2.2 Facts
        2.2.3 Temporal Extraction and Edge Invalidation
    2.3 Communities
3 Memory Retrieval
    3.1 Search
    3.2 Reranker
4 Experiments
    4.1 Choice of models
    4.2 Deep Memory Retrieval (DMR)
    4.3 LongMemEval (LME)
        4.3.1 LongMemEval and MemGPT
        4.3.2 LongMemEval results
5 Conclusion
6 Appendix
    6.1 Graph Construction Prompts
        6.1.1 Entity Extraction
        6.1.2 Entity Resolution
        6.1.3 Fact Extraction
        6.1.4 Fact Resolution
        6.1.5 Temporal Extraction
References
## Contents
- 1 Introduction
- 2 Knowledge Graph Construction
  - 2.1 Episodes
  - 2.2 Semantic Entities and Facts
    - 2.2.1 Entities
    - 2.2.2 Facts
    - 2.2.3 Temporal Extraction and Edge Invalidation
  - 2.3 Communities
- 3 Memory Retrieval
  - 3.1 Search
  - 3.2 Reranker
- 4 Experiments
  - 4.1 Choice of models
  - 4.2 Deep Memory Retrieval (DMR)
  - 4.3 LongMemEval (LME)
    - 4.3.1 LongMemEval and MemGPT
    - 4.3.2 LongMemEval results
- 5 Conclusion
- 6 Appendix
  - 6.1 Graph Construction Prompts
    - 6.1.1 Entity Extraction
    - 6.1.2 Entity Resolution
    - 6.1.3 Fact Extraction
    - 6.1.4 Fact Resolution
    - 6.1.5 Temporal Extraction
- References

## Abstract

Abstract We introduce Zep, a novel memory layer service for AI agents that outperforms the current state-of-the-art system, MemGPT, in the Deep Memory Retrieval (DMR) benchmark. Additionally, Zep excels in more comprehensive and challenging evaluations than DMR that better reflect real-world enterprise use cases. While existing retrieval-augmented generation (RAG) frameworks for large language model (LLM)-based agents are limited to static document retrieval, enterprise applications demand dynamic knowledge integration from diverse sources including ongoing conversations and business data. Zep addresses this fundamental limitation through its core component Graphiti—a temporally-aware knowledge graph engine that dynamically synthesizes both unstructured conversational data and structured business data while maintaining historical relationships. In the DMR benchmark, which the MemGPT team established as their primary evaluation metric, Zep demonstrates superior performance (94.8% vs 93.4%). Beyond DMR, Zep’s capabilities are further validated through the more challenging LongMemEval benchmark, which better reflects enterprise use cases through complex temporal reasoning tasks. In this evaluation, Zep achieves substantial results with accuracy improvements of up to 18.5% while simultaneously reducing response latency by 90% compared to baseline implementations. These results are particularly pronounced in enterprise-critical tasks such as cross-session information synthesis and long-term context maintenance, demonstrating Zep’s effectiveness for deployment in real-world applications.

## 1 Introduction

The impact of transformer-based large language models (LLMs) on industry and research communities has garnered significant attention in recent years [1]. A major application of LLMs has been the development of chat-based agents. However, these agents’ capabilities are limited by the LLMs’ context windows, effective context utilization, and knowledge gained during pre-training. Consequently, additional context is required to provide out-of-domain (OOD) knowledge and reduce hallucinations.

Retrieval-Augmented Generation (RAG) has emerged as a key area of interest in LLM-based applications. RAG leverages Information Retrieval (IR) techniques pioneered over the last fifty years[2] to supply necessary domain knowledge to LLMs.

Current approaches using RAG have focused on broad domain knowledge and largely static corpora—that is, document contents added to a corpus seldom change. For agents to become pervasive in our daily lives, autonomously solving problems from trivial to highly complex, they will need access to a large corpus of continuously evolving data from users’ interactions with the agent, along with related business and world data. We view empowering agents with this broad and dynamic "memory" as a crucial building block to actualize this vision, and we argue that current RAG approaches are unsuitable for this future. Since entire conversation histories, business datasets, and other domain-specific content cannot fit effectively inside LLM context windows, new approaches need to be developed for agent memory. Adding memory to LLM-powered agents isn’t a new idea—this concept has been explored previously in MemGPT [3].

Recently, Knowledge Graphs (KGs) have been employed to enhance RAG architectures to address many of the shortcomings of traditional IR techniques[4]. In this paper, we introduce Zep[5], a memory layer service powered by Graphiti[6], a dynamic, temporally-aware knowledge graph engine. Zep ingests and synthesizes both unstructured message data and structured business data. The Graphiti KG engine dynamically updates the knowledge graph with new information in a non-lossy manner, maintaining a timeline of facts and relationships, including their periods of validity. This approach enables the knowledge graph to represent a complex, evolving world.

As Zep is a production system, we’ve focused heavily on the accuracy, latency, and scalability of its memory retrieval mechanisms. We evaluate these mechanisms’ efficacy using two existing benchmarks: a Deep Memory Retrieval task (DMR) from MemGPT[3], as well as the LongMemEval benchmark[7].

## 2 Knowledge Graph Construction

In Zep, memory is powered by a temporally-aware dynamic knowledge graph $\mathcal{G}=(\mathcal{N},\mathcal{E},\phi)$, where $\mathcal{N}$ represents nodes, $\mathcal{E}$ represents edges, and $\phi:\mathcal{E}\to\mathcal{N}\times\mathcal{N}$ represents a formal incidence function. This graph comprises three hierarchical tiers of subgraphs: an episode subgraph, a semantic entity subgraph, and a community subgraph.

- •
Episode Subgraph $\mathcal{G}_{e}$: Episodic nodes (episodes), $n_{i}\in\mathcal{N}_{e}$, contain raw input data in the form of messages, text, or JSON. Episodes serve as a non-lossy data store from which semantic entities and relations are extracted. Episodic edges, $e_{i}\in\mathcal{E}_{e}\subseteq\phi^{*}(\mathcal{N}_{e}\times\mathcal{N}_{s})$, connect episodes to their referenced semantic entities.
- •
Semantic Entity Subgraph $\mathcal{G}_{s}$: The semantic entity subgraph builds upon the episode subgraph. Entity nodes (entities), $n_{i}\in\mathcal{N}_{s}$, represent entities extracted from episodes and resolved with existing graph entities. Entity edges (semantic edges), $e_{i}\in\mathcal{E}_{s}\subseteq\phi^{*}(\mathcal{N}_{s}\times\mathcal{N}_{s})$, represent relationships between entities extracted from episodes.
- •
Community Subgraph $\mathcal{G}_{c}$: The community subgraph forms the highest level of Zep’s knowledge graph. Community nodes (communities), $n_{i}\in\mathcal{N}_{c}$, represent clusters of strongly connected entities. Communities contain high-level summarizations of these clusters and represent a more comprehensive, interconnected view of $\mathcal{G}_{s}$’s structure. Community edges, $e_{i}\in\mathcal{E}_{c}\subseteq\phi^{*}(\mathcal{N}_{c}\times\mathcal{N}_{s})$, connect communities to their entity members.

The dual storage of both raw episodic data and derived semantic entity information mirrors psychological models of human memory. These models distinguish between episodic memory, which represents distinct events, and semantic memory, which captures associations between concepts and their meanings [8]. This approach enables LLM agents using Zep to develop more sophisticated and nuanced memory structures that better align with our understanding of human memory systems. Knowledge graphs provide an effective medium for representing these memory structures, and our implementation of distinct episodic and semantic subgraphs draws from similar approaches in AriGraph [9].

Our use of community nodes to represent high-level structures and domain concepts builds upon work from GraphRAG [4], enabling a more comprehensive global understanding of the domain. The resulting hierarchical organization—from episodes to facts to entities to communities—extends existing hierarchical RAG strategies [10][11].

### 2.1 Episodes

Zep’s graph construction begins with the ingestion of raw data units called Episodes. Episodes can be one of three core types: message, text, or JSON. While each type requires specific handling during graph construction, this paper focuses on the message type, as our experiments center on conversation memory. In our context, a message consists of relatively short text (several messages can fit within an LLM context window) along with the associated actor who produced the utterance.

Each message includes a reference timestamp $t_{\text{ref}}$ indicating when the message was sent. This temporal information enables Zep to accurately identify and extract relative or partial dates mentioned in the message content (e.g., "next Thursday," "in two weeks," or "last summer"). Zep implements a bi-temporal model, where timeline $T$ represents the chronological ordering of events, and timeline $T^{\prime}$ represents the transactional order of Zep’s data ingestion. While the $T^{\prime}$ timeline serves the traditional purpose of database auditing, the $T$ timeline provides an additional dimension for modeling the dynamic nature of conversational data and memory. This bi-temporal approach represents a novel advancement in LLM-based knowledge graph construction and underlies much of Zep’s unique capabilities compared to previous graph-based RAG proposals.

The episodic edges, $\mathcal{E}_{e}$, connect episodes to their extracted entity nodes. Episodes and their derived semantic edges maintain bidirectional indices that track the relationships between edges and their source episodes. This design reinforces the non-lossy nature of Graphiti’s episodic subgraph by enabling both forward and backward traversal: semantic artifacts can be traced to their sources for citation or quotation, while episodes can quickly retrieve their relevant entities and facts. While these connections are not directly examined in this paper’s experiments, they will be explored in future work.

### 2.2 Semantic Entities and Facts

#### 2.2.1 Entities

ntity extraction represents the initial phase of episode processing. During ingestion, the system processes both the current message content and the last $n$ messages to provide context for named entity recognition. For this paper and in Zep’s general implementation, $n=4$, providing two complete conversation turns for context evaluation. Given our focus on message processing, the speaker is automatically extracted as an entity. Following initial entity extraction, we employ a reflection technique inspired by reflexion[12] to minimize hallucinations and enhance extraction coverage. The system also extracts an entity summary from the episode to facilitate subsequent entity resolution and retrieval operations.

After extraction, the system embeds each entity name into a 1024-dimensional vector space. This embedding enables the retrieval of similar nodes through cosine similarity search across existing graph entity nodes. The system also performs a separate full-text search on existing entity names and summaries to identify additional candidate nodes. These candidate nodes, together with the episode context, are then processed through an LLM using our entity resolution prompt. When the system identifies a duplicate entity, it generates an updated name and summary.

Following entity extraction and resolution, the system incorporates the data into the knowledge graph using predefined Cypher queries. We chose this approach over LLM-generated database queries to ensure consistent schema formats and reduce the potential for hallucinations.

Selected prompts for graph construction are provided in the appendix.

#### 2.2.2 Facts

or each fact containing its key predicate. Importantly, the same fact can be extracted multiple times between different entities, enabling Graphiti to model complex multi-entity facts through an implementation of hyper-edges.

Following extraction, the system generates embeddings for facts in preparation for graph integration. The system performs edge deduplication through a process similar to entity resolution. The hybrid search for relevant edges is constrained to edges existing between the same entity pairs as the proposed new edge. This constraint not only prevents erroneous combinations of similar edges between different entities but also significantly reduces the computational complexity of the deduplication process by limiting the search space to a subset of edges relevant to the specific entity pair.

#### 2.2.3 Temporal Extraction and Edge Invalidation

A key differentiating feature of Graphiti compared to other knowledge graph engines is its capacity to manage dynamic information updates through temporal extraction and edge invalidation processes.

The system extracts temporal information about facts from the episode context using $t_{\text{ref}}$. This enables accurate extraction and datetime representation of both absolute timestamps (e.g., "Alan Turing was born on June 23, 1912") and relative timestamps (e.g., "I started my new job two weeks ago"). Consistent with our bi-temporal modeling approach, the system tracks four timestamps: $t^{\prime}\text{created}$ and $t^{\prime}\text{expired}\in T^{\prime}$ monitor when facts are created or invalidated in the system, while $t_{\text{valid}}$ and $t_{\text{invalid}}\in T$ track the temporal range during which facts held true. These temporal data points are stored on edges alongside other fact information.

The introduction of new edges can invalidate existing edges in the database. The system employs an LLM to compare new edges against semantically related existing edges to identify potential contradictions. When the system identifies temporally overlapping contradictions, it invalidates the affected edges by setting their $t_{\text{invalid}}$ to the $t_{\text{valid}}$ of the invalidating edge. Following the transactional timeline $T^{\prime}$, Graphiti consistently prioritizes new information when determining edge invalidation.

This comprehensive approach enables the dynamic addition of data to Graphiti as conversations evolve, while maintaining both current relationship states and historical records of relationship evolution over time.

### 2.3 Communities

After establishing the episodic and semantic subgraphs, the system constructs the community subgraph through community detection. While our community detection approach builds upon the technique described in GraphRAG[4], we employ a label propagation algorithm [13] rather than the Leiden algorithm [14]. This choice was influenced by label propagation’s straightforward dynamic extension, which enables the system to maintain accurate community representations for longer periods as new data enters the graph, delaying the need for complete community refreshes.

The dynamic extension implements the logic of a single recursive step in label propagation. When the system adds a new entity node $n_{i}\in\mathcal{N}_{s}$ to the graph, it surveys the communities of neighboring nodes. The system assigns the new node to the community held by the plurality of its neighbors, then updates the community summary and graph accordingly. While this dynamic updating enables efficient community extension as data flows into the system, the resulting communities gradually diverge from those that would be generated by a complete label propagation run. Therefore, periodic community refreshes remain necessary. However, this dynamic updating strategy provides a practical heuristic that significantly reduces latency and LLM inference costs.

Following [4], our community nodes contain summaries derived through an iterative map-reduce-style summarization of member nodes. However, our retrieval methods differ substantially from GraphRAG’s map-reduce approach [4]. To support our retrieval methodology, we generate community names containing key terms and relevant subjects from the community summaries. These names are embedded and stored to enable cosine similarity searches.

## 3 Memory Retrieval

The memory retrieval system in Zep provides powerful, complex, and highly configurable functionality. At a high level, the Zep graph search API implements a function $f:S\to S$ that accepts a text-string query $\alpha\in S$ as input and returns a text-string context $\beta\in S$ as output. The output $\beta$ contains formatted data from nodes and edges required for an LLM agent to generate an accurate response to query $\alpha$. The process $f(\alpha)\to\beta$ comprises three distinct steps:

- •
Search ($\varphi$): The process begins by identifying candidate nodes and edges potentially containing relevant information. While Zep employs multiple distinct search methods, the overall search function can be represented as $\varphi:S\to\mathcal{E}_{s}^{n}\times\mathcal{N}_{s}^{n}\times\mathcal{N}_{c}^
{n}$. Thus, $\varphi$ transforms a query into a 3-tuple containing lists of semantic edges, entity nodes, and community nodes—the three graph types containing relevant textual information.
- •
Reranker ($\rho$): The second step reorders search results. A reranker function or model accepts a list of search results and produces a reordered version of those results: $\rho:{\varphi(\alpha),...}\to\mathcal{E}_{s}^{n}\times\mathcal{N}_{s}^{n}
\times\mathcal{N}_{c}^{n}$.
- •
Constructor ($\chi$): The final step, the constructor, transforms the relevant nodes and edges into text context: $\chi:\mathcal{E}_{s}^{n}\times\mathcal{N}_{s}^{n}\times\mathcal{N}c^{n}\to S$. For each $e_{i}\in\mathcal{E}s$, $\chi$ returns the fact and $t\text{valid},t\text{invalid}$ fields; for each $n_{i}\in\mathcal{N}_{s}$, the name and summary fields; and for each $n_{i}\in\mathcal{N}_{c}$, the summary field.

With these definitions established, we can express $f$ as a composition of these three components: $f(\alpha)=\chi(\rho(\varphi(\alpha)))=\beta$.

Sample context string template:

FACTS and ENTITIES represent relevant context to the current conversation.
These are the most relevant facts and their valid date ranges. If the fact is about an event, the event takes place during this time.
format: FACT (Date range: from - to)
<FACTS>
{facts}
</FACTS>
These are the most relevant entities
ENTITY_NAME: entity summary
<ENTITIES>
{entities}
</ENTITIES>

### 3.1 Search

Zep implements three search functions: cosine semantic similarity search $(\varphi_{\text{cos}})$, Okapi BM25 full-text search $(\varphi_{\text{bm25}})$, and breadth-first search $(\varphi_{\text{bfs}})$. The first two functions utilize Neo4j’s implementation of Lucene [15][16]. Each search function offers distinct capabilities in identifying relevant documents, and together they provide comprehensive coverage of candidate results before reranking. The search field varies across the three object types: for $\mathcal{E}_{s}$, we search the fact field; for $\mathcal{N}_{s}$, the entity name; and for $\mathcal{N}_{c}$, the community name, which comprises relevant keywords and phrases covered in the community. While developed independently, our community search approach parallels the high-level key search methodology in LightRAG [17]. The hybridization of LightRAG’s approach with graph-based systems like Graphiti presents a promising direction for future research.

While cosine similarity and full-text search methodologies are well-established in RAG [18], breadth-first search over knowledge graphs has received limited attention in the RAG domain, with notable exceptions in graph-based RAG systems such as AriGraph [9] and Distill-SynthKG [19]. In Graphiti, the breadth-first search enhances initial search results by identifying additional nodes and edges within $n$-hops. Moreover, $\varphi_{\text{bfs}}$ can accept nodes as parameters for the search, enabling greater control over the search function. This functionality proves particularly valuable when using recent episodes as seeds for the breadth-first search, allowing the system to incorporate recently mentioned entities and relationships into the retrieved context.

The three search methods each target different aspects of similarity: full-text search identifies word similarities, cosine similarity captures semantic similarities, and breadth-first search reveals contextual similarities—where nodes and edges closer in the graph appear in more similar conversational contexts. This multi-faceted approach to candidate result identification maximizes the likelihood of discovering optimal context.

### 3.2 Reranker

While the initial search methods aim to achieve high recall, rerankers serve to increase precision by prioritizing the most relevant results. Zep supports existing reranking approaches such as Reciprocal Rank Fusion (RRF) [20] and Maximal Marginal Relevance (MMR) [21]. Additionally, Zep implements a graph-based episode-mentions reranker that prioritizes results based on the frequency of entity or fact mentions within a conversation, enabling a system where frequently referenced information becomes more readily accessible. The system also includes a node distance reranker that reorders results based on their graph distance from a designated centroid node, providing context localized to specific areas of the knowledge graph. The system’s most sophisticated reranking capability employs cross-encoders—LLMs that generate relevance scores by evaluating nodes and edges against queries using cross-attention, though this approach incurs the highest computational cost.

## 5 Conclusion

We have introduced Zep, a graph-based approach to LLM memory that incorporates semantic and episodic memory alongside entity and community summaries. Our evaluations demonstrate that Zep achieves state-of-the-art performance on existing memory benchmarks while reducing token costs and operating at significantly lower latencies.

The results achieved with Graphiti and Zep, while impressive, likely represent only initial advances in graph-based memory systems. Multiple research avenues could build upon these frameworks, including integration of other GraphRAG approaches into the Zep paradigm and novel extensions of our work.

Research has already demonstrated the value of fine-tuned models for LLM-based entity and edge extraction within the GraphRAG paradigm, improving accuracy while reducing costs and latency [19][25]. Similar models fine-tuned for Graphiti prompts may enhance knowledge extraction, particularly for complex conversations. Additionally, while current research on LLM-generated knowledge graphs has primarily operated without formal ontologies [9][4][17][19][26], domain-specific ontologies present significant potential. Graph ontologies, foundational in pre-LLM knowledge graph work, warrant further exploration within the Graphiti framework.

Our search for suitable memory benchmarks revealed limited options, with existing benchmarks often lacking robustness and complexity, frequently defaulting to simple needle-in-a-haystack fact-retrieval questions [3]. The field requires additional memory benchmarks, particularly those reflecting business applications like customer experience tasks, to effectively evaluate and differentiate memory approaches. Notably, no existing benchmarks adequately assess Zep’s capability to process and synthesize conversation history with structured business data. While Zep focuses on LLM memory, its traditional RAG capabilities should be evaluated against established benchmarks such as those in [17], [27], and [28].

Current literature on LLM memory and RAG systems insufficiently addresses production system scalability in terms of cost and latency. We have included latency benchmarks for our retrieval mechanisms to begin addressing this gap, following the example set by LightRAG’s authors in prioritizing these metrics.

## 6 Appendix

### 6.1 Graph Construction Prompts

#### 6.1.1 Entity Extraction

<PREVIOUS MESSAGES>
{previous_messages}
</PREVIOUS MESSAGES>
<CURRENT MESSAGE>
{current_message}
</CURRENT MESSAGE>
Given the above conversation, extract entity nodes from the
CURRENT MESSAGE that are explicitly or implicitly mentioned:
Guidelines:
1. ALWAYS extract the speaker/actor as the first node. The speaker
is the part before the colon in each line of dialogue.
2. Extract other significant entities, concepts, or actors mentioned
in the CURRENT MESSAGE.
3. DO NOT create nodes for relationships or actions.
4. DO NOT create nodes for temporal information like dates, times
or years (these will be added to edges later).
5. Be as explicit as possible in your node names, using full names.
6. DO NOT extract entities mentioned only

#### 6.1.2 Entity Resolution

<PREVIOUS MESSAGES>
{previous_messages}
</PREVIOUS MESSAGES>
<CURRENT MESSAGE>
{current_message}
</CURRENT MESSAGE>
<EXISTING NODES>
{existing_nodes}
</EXISTING NODES>
Given the above EXISTING NODES, MESSAGE, and PREVIOUS MESSAGES. Determine if the NEW NODE extracted from the conversation
is a duplicate entity of one of the EXISTING NODES.
<NEW NODE>
{new_node}
</NEW NODE>
Task:
1. If the New Node represents the same entity as any node in Existing Nodes,
return ’is_duplicate: true’ in the
response. Otherwise, return ’is_duplicate: false’
2. If is_duplicate is true, also return the uuid of the existing node in the
response
3. If is_duplicate is true, return a name for the node that is the most complete
full name.
Guidelines:
1. Use both the name and summary of nodes to determine if the entities are
duplicates, duplicate nodes may have different names

#### 6.1.3 Fact Extraction

<PREVIOUS MESSAGES>
{previous_messages}
</PREVIOUS MESSAGES>
<CURRENT MESSAGE>
{current_message}
</CURRENT MESSAGE>
<ENTITIES>
{entities}
</ENTITIES>
Given the above MESSAGES and ENTITIES, extract all facts pertaining to the
listed ENTITIES from the CURRENT MESSAGE.
Guidelines:
1. Extract facts only between the provided entities.
2. Each fact should represent a clear relationship between two DISTINCT nodes.
3. The relation_type should be a concise, all-caps description of the
fact (e.g., LOVES, IS_FRIENDS_WITH, WORKS_FOR).
4. Provide a more detailed fact containing all relevant information.
5. Consider temporal aspects of relationships when relevant.

#### 6.1.4 Fact Resolution

Given the following context, determine whether the New Edge represents any
of the edges in the list of Existing Edges.
<EXISTING EDGES>
{existing_edges}
</EXISTING EDGES>
<NEW EDGE>
{new_edge}
</NEW EDGE>
Task:
1. If the New Edges represents the same factual information as any edge
in Existing Edges, return ’is_duplicate: true’ in the response.
Otherwise, return ’is_duplicate: false’
2. If is_duplicate is true, also return the uuid of the existing edge in
the response
Guidelines:
1. The facts do not need to be completely identical to be duplicates,
they just need to express the same information.

#### 6.1.5 Temporal Extraction

<PREVIOUS MESSAGES>
{previous_messages}
</PREVIOUS MESSAGES>
<CURRENT MESSAGE>
{current_message}
</CURRENT MESSAGE>
<REFERENCE TIMESTAMP>
{reference_timestamp}
</REFERENCE TIMESTAMP>
<FACT>
{fact}
</FACT>
IMPORTANT: Only extract time information if it is part of the provided fact.
Otherwise ignore the time mentioned.
Make sure to do your best to determine the dates if only the relative time is
mentioned. (eg 10 years ago, 2 mins ago) based on the provided reference
timestamp
If the relationship is not of spanning nature, but you are still able to
determine the dates, set the valid_at only.
Definitions:
- valid_at: The date and time when the relationship described by the
edge fact became true or was established.
- invalid_at: The date and time when the relationship described by the
edge fact stopped being true or ended.
Task:
Analyze the conversation and determine if there are dates that are part
of the edge fact. Only set dates if they explicitly relate to the
formation or alteration of the relationship itself.
Guidelines:
1. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.SSSSSSZ) for datetimes.
2. Use the reference timestamp as the current time when determining
the valid_at and invalid_at dates.
3. If the fact is written in the present tense, use the Reference
Timestamp for the valid_at date
4. If no temporal information is found that establishes or changes the
relationship, leave the fields as null.
5. Do not infer dates from related events. Only use dates that are
directly stated to establish or change the relationship.
6. For relative time mentions directly related to the relationship, calculate the
actual datetime based on the reference timestamp.
7. If only a date is mentioned without a specific time, use 00:00:00 (midnight) for that date.
8. If only year is mentioned, use January 1st of that year at 00:00:00.
9. Always include the time zone offset (use Z for UTC if no specific
time zone is mentioned).
