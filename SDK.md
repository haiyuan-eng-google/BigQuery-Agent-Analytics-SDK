![][image1] SDK for BigQuery Agent Analytics  
---

| Useful information Status: Under Review  Turbo Launch: \<add your link\> Last updated: Feb 9, 2026 Authors: [Sandeep Karmarkar](mailto:karmarkar@google.com),  Collaborators: [Haiyuan Cao](mailto:haiyuan@google.com) Short link:   Links to other materials Design Doc: \<add your link\> User Guide: \<add your link\> Content  [Background](#background) [Market Opportunity](#market-opportunity) [Why BigQuery?](#why-bigquery?) [Current Foundation](#current-foundation) [The Goal: The BigQuery Agent Analytics SDK](#the-goal:-the-bigquery-agent-analytics-sdk) [Target Users](#target-users) [Critical User Journeys](#critical-user-journeys) [CUJ 1: Debugging of a Specific Session (The "Trace" Journey)](#cuj-1:-debugging-of-a-specific-session-\(the-"trace"-journey\)) [CUJ 2: Offline Evaluation with a "Golden Dataset"](#cuj-2:-offline-evaluation-with-a-"golden-dataset") [CUJ 3: Online Evaluation (Without a Golden Dataset)](#cuj-3:-online-evaluation-\(without-a-golden-dataset\)) [CUJ 4: Feedback Loop & Curation](#cuj-4:-feedback-loop-&-curation) [Functional Requirements](#functional-requirements) [Desired Language Support](#desired-language-support) [Classes and Methods](#classes-and-methods) [Desired Backend Support](#desired-backend-support) [Competitive Offerings](#competitive-offerings)   | Approval Log Role Username Date GM/Delegate yyyy-mm-dd Eng Lead yyyy-mm-dd PM Lead yyyy-mm-dd SRE Lead yyyy-mm-dd UX Lead yyyy-mm-dd Changelog Editor Comments Date \<editor1@\> V1 approved yyyy-mm-dd \<editor2@\> Additions to V1 approved yyyy-mm-dd  |
| :---- | :---- |

# **Background** {#background}

Agent observability is the critical missing link in the transition from "GenAI prototype" to "Production Agent." While traditional application monitoring focuses on system health (latency, error rates, CPU), agent observability requires a fundamental shift in perspective to focus on the **quality and logic of the reasoning engine**.

As noted in the [LangChain blog on Agent Observability](https://www.langchain.com/conceptual-guides/agent-observability-powers-agent-evaluation), observing agents poses unique challenges compared to standard software applications:

1. **Non-Determinism & Hallucination:** Unlike traditional code where Input A always equals Output B, agents are probabilistic. Observability must track not just *if* it worked, but *what* it said, requiring semantic evaluation rather than simple binary error checks.  
2. **Complex, Multi-Step Traces:** Agents often invoke tools, query databases, and loop through reasoning steps. A simple stack trace is insufficient; developers need to visualize the full Directed Acyclic Graph (DAG) of the conversation to understand at which specific step the logic diverged (e.g., did the tool fail, or did the LLM misinterpret the tool's output?).  
3. **Feedback Loops as a Requirement:** In traditional software, logs are for debugging. In agents, logs are **training data**. Observability must seamlessly feed into dataset curation to improve the model or the prompt strategies (few-shot examples).

## **Market Opportunity** {#market-opportunity}

This is a rapidly exploding market. As enterprises move past the initial "hype" phase, the demand for **LLMOps** and **AI Observability** solutions is projected to grow significantly, with analysts estimating the AI observability market to reach multibillion-dollar valuations by 2030 \[[e.g.](https://www.mordorintelligence.com/industry-reports/agentic-artificial-intelligence-monitoring-analytics-and-observability-tools-market#:~:text=Agentic%20AI%20Monitoring%2C%20Analytics%2C%20And%20Observability%20Tools%20Market%20Analysis%20by,assurance%2C%20especially%20in%20regulated%20verticals.])\] Companies in this space like [Arize AI](https://arize.com/), [LangSmith](https://smith.langchain.com/), and [Weights & Biases](https://wandb.ai/site/) are gaining traction and we have customers evaluating those. Clickhouse also recently [acquired](https://clickhouse.com/blog/clickhouse-acquires-langfuse-open-source-llm-observability) Langfuse (another agent observability vendor). 

## **Why BigQuery?**  {#why-bigquery?}

We believe BigQuery brings unique scale and capabilities that make it the right platform for Agent observability \- 

1. **Scalable "llm-as-a-judge" (Zero-ETL Evaluation) :** Agent evaluations, feedback loops go beyond traditional metrics and errors. Agent developers are looking to deploy agents to help score /rate the conversations, identify patterns of similar interactions, identify drifts from golden training sets etc.  BigQuery , with its AI functions , Vector/Hybrid search capabilities and native scalable architecture is uniquely positioned to accomplish these goals for users in a scalable and cost effective way.  Competitors like Arize end up making LLM calls to external end points for every session / log event making it hard to use at scale.   
2. **Support for multimodal data :** Modern agents interact with images and audio. BigQuery’s multimodal Tables and integration with BQ AI capabilitiesI allow for seamless logging and analysis of multimodal data (blobs) without complex pre-processing, a capability that many text-first observability tools lack.  
3. **Data Sovereignty & Security:**  Agent interactions can capture sensitive information. BigQuery offers unique and fine-grained capabilities to mask and protect information all the way to row and column level. 

While GCP is investing in a managed Agent observability solution \[through cloud logging and cloud tracing \] , BigQuery will cater to larger enterprises, ISVs who are building their own evaluation , observability solutions or already have investments in alternative observability platforms (like DataDog)

## **Current Foundation**  {#current-foundation}

We have validated the market demand and  laid the groundwork for this ecosystem by releasing ingestion plugins that stream agent traces directly into BigQuery

*  [***Google Cloud ADK BigQuery Plugin:***](https://google.github.io/adk-docs/observability/bigquery-agent-analytics/) **For high-performance, asynchronous logging of Vertex AI and custom agents.*   
* [**LangChain BigQuery Callback:**](https://python.langchain.com/docs/integrations/callbacks/google_bigquery/) For native integration with the open-source LangChain framework.

Since launch  (\<2 months) 100+ customers have used these plugins and have ingested \>2M rows.  Customers interested in preview include \- ISVs like AppsFlyers and large customers like Comcast, Globo etc. 

## **The Goal: The BigQuery Agent Analytics SDK**  {#the-goal:-the-bigquery-agent-analytics-sdk}

While these plugins solve the *storage* problem, the *analysis* problem remains. The raw data stored in BigQuery requires advanced SQL knowledge to use it effectively.

With this PRD, we propose building a **Python SDK** that serves as the consumption layer for this data. This SDK will allow agent developers to: 

* **Reconstruct and Visualize Sessions**: Instantly render conversation traces in Notebooks without writing UNNEST SQL queries.  
* **Run Offline/Online Evaluations**: Define metrics in Python that execute as high-performance BQML jobs in the background.  
* **Close the Loop**: Curate high-quality examples from production logs to create "Golden Datasets" for regression testing.


Critically, this SDK **eliminates BigQuery-specific learning curves**. It abstracts away complex SQL syntax, Table definitions, and BQML mechanics, exposing clean, intuitive Python interfaces that AI Engineers already understand.

# **Target Users** {#target-users}

Agent developers are the intended target users with this SDK. We do not expect these users to be advanced BigQuery users .  We do expect these users to be advanced Python developers. 

# **Critical User Journeys** {#critical-user-journeys}

### **CUJ 1: Debugging of a Specific Session (The "Trace" Journey)** {#cuj-1:-debugging-of-a-specific-session-(the-"trace"-journey)}

**Goal:** Quickly investigate a report of a failed conversation (e.g., "The bot hallucinated on session ID `12345`").

**Journey with SDK:**

1. Users initiate their analysis within a Jupyter Notebook or preferred local IDE.  
2. They initialize the SDK, retrieving a specific trace, `e.g., trace = client.get_trace("12345")`. `# Automatically resolves GCS-offloaded payloads if configured (e.g. gcs_bucket_name provided in Client)`  
3. They invoke the visualization command: `trace.render()`.  
4. **Result:** The SDK seamlessly processes the raw logs and presents a compelling, hierarchical tree (DAG) of the entire interaction: User Input  →  Agent Thought →  Tool Call (Search) → Tool Output →  Final Agent Response.   
5. *Multimodal Capability:* Should the user have provided an image, the SDK dynamically generates a signed URL, enabling the inline display of the visual data within the notebook, thereby allowing the engineer to instantly confirm if the model misinterpreted the input.

*Enhancements to consider include filtering by timestamp, associated tags, observed failures, and sentiment analysis.*

### **CUJ 2: Offline Evaluation with a "Golden Dataset"** {#cuj-2:-offline-evaluation-with-a-"golden-dataset"}

**Goal:** Ensure that a newly developed agent version can accurately address established "Golden" questions before its deployment to a production environment.

**Journey with SDK:**

1. The user loads their definitive "Golden Set" (comprising Questions \+ Ideal Answers) into a BigQuery table via the SDK (or references an existing BigQuery dataset).  
2. The user then defines a suitable evaluator:  
   1. Code-Based: Deterministic, user-defined functions, such as measuring response latency or turn count.   
      1. The Plugin provides a suite of pre-built code evaluators.  
   2. LLM as a Judge: The user articulates a quality metric (e.g., “Correctness” or “Hallucination”) and provides specific LLM prompts to guide the automated evaluation process.   
      1. The Plugin supplies numerous built-in LLM evaluators with proven, predefined prompts.  
3. The user executes the evaluation test:

e.g. `evaluate(golden_set="v1", table="test_table", data-select_filters =”key:value”, evaluator=”my_evaluator”)`. 

4. The SDK leverages the underlying BigQuery capabilities to execute the scoring against the golden set and subsequently generates a comprehensive report, which can also be persisted in BigQuery.

### **CUJ 3: Online Evaluation (Without a Golden Dataset)** {#cuj-3:-online-evaluation-(without-a-golden-dataset)}

**Goal:** Similar to offline evaluation, users aim to assess the performance of a currently deployed agent in a live environment.  
**Journey with SDK:**

1. The user defines an appropriate evaluator:  
   1. Code-Based: Deterministic user-defined functions, for example, measuring based on turn count or response latencies. Since there is no reference golden set, the user must define an “acceptable” performance range.  
      1. The Plugin offers a selection of pre-built code evaluators.  
   2. LLM as a Judge: The user specifies metrics like “Correctness,” “Hallucination,” “Sentiment,” or “Works-as-Expected,” and provides the necessary LLM prompts to guide the evaluation.  
      1. The Plugin includes several built-in LLM evaluators with predefined prompts.  
2. The user initiates an evaluation test:

e.g. `evaluate(dataset=yesterday_traffic, evaluator=”my_evaluator”)`

3. The SDK leverages BigQuery capabilities to execute the evaluation and generates a report that can be stored in BigQuery for ongoing monitoring.

### 

### **CUJ 4: Feedback Loop & Curation** {#cuj-4:-feedback-loop-&-curation}

**Goal:** Identify key opportunities to enhance the agent, which includes refining the golden dataset.

**Journey with SDK:**

1. **Drift Detection** The user submits the golden question dataset and recent production traces, along with required filters (e.g., time range or agent ID). The SDK then compares the golden dataset against actual questions, reporting the percentage of questions adequately covered. Clustered visualizations can be provided upon request.  
2. **Question Distribution** The user supplies production traces and desired filters, requesting distributions of questions categorized as: *frequently\_asked*, *frequently\_unaswered*, *auto\_group\_using\_semantics*, or specific semantic groupings defined via natural language (e.g., “onboarding related,” “PTO related,” “Salary related,” or “legal”). The SDK returns the calculated distribution.

# **Functional Requirements** {#functional-requirements}

## **Desired Language Support** {#desired-language-support}

* **Python-First Experience:** The primary interface must be a Python SDK to align with the existing workflows of AI Engineers and Data Scientists.  
* **Notebook Compatibility:** The SDK must be optimized for interactive environments like Jupyter and Colab, supporting inline visualizations and rich media rendering.  
* **SQL Abstraction:** The SDK must abstract complex BigQuery SQL (e.g., `UNNEST` operations, JSON parsing) into clean Pythonic calls. We should reuse dataframes abstraction provided by BigFrames 

## **Classes and Methods** {#classes-and-methods}

* **Client Initialization:**  
  * `Client(project_id, dataset_id, table_id, location="us-central1", gcs_bucket_name=None, verify_schema=True)` : Initialize connection to the BigQuery telemetry store.  
* **Trace Reconstruction:**  
  * `get_trace(trace_id)`: Fetches all spans associated with a specific session  
  * `list_traces(filter_criteria)`: Enables discovery of traces based on tags, timestamps, performance metrics.  
* **Advanced filters**   
  * Filter traces using errors (any or specific), sentiments or any other LLM filter.   
* **Visualization:**  
  * `trace.render()`: Generates a hierarchical Directed Acyclic Graph (DAG) or tree view of the agent’s reasoning steps (Thought → Tool Call → Response).  
  * Support for rendering multimodal content (images, audio) via signed URLs.  
* **Evaluation Engine:**  
  * `evaluate(dataset, filters, evaluator, ...)`: High-level method to trigger batch scoring jobs.  
    * Support filtering dataset using filters like timestamp, agent\_id,  custom labels (e.g. experiment\_id)  
  * Pre-built Evaluator Classes:  
    * `CodeEvaluator`: For deterministic metrics like latency and turn count.  
    * `LLMAsJudge`: For semantic metrics like "Correctness" or "Hallucination" using predefined or custom prompts.  
  * Support evaluation against golden dataset   
  * \[P1\] Support evaluation of two filtered views within the same (of different dataset ) e..g. Two experiments.   
* **Feedback and curation:**  
  * `drift_detection(dataset, filters, golden_dataset,...)`: Method to detect drifts from golden dataset   
  * `deep_analysis(dataset, filters, configuration,...)`: Method to get deep analysis based on configuration.   
    * E.g. question distribution based on categories like *frequently\_asked*, *frequently\_unaswered*, *auto\_group\_using\_semantics*, or specific semantic groupings defined via natural language

## **Desired Backend Support** {#desired-backend-support}

* **BigFrames \-**  for scalable pandas.   
* **Scalable BQML/AI Functions:** The backend must leverage BigQuery’s native AI functions to execute "LLM-as-a-judge" evaluations at scale without egressing data to external endpoints.  
* **Vector & Hybrid Search:** Native support for semantic clustering and drift detection between production logs and golden datasets.  
* **Security & Sovereignty:** Integration with BigQuery's row and column-level security to mask sensitive information captured during agent interactions.  
* **Multimodal Tables:** Native handling of image and audio blobs to support observability for multimodal agents.

# **Competitive Offerings** {#competitive-offerings}

|  | LangFuse | LangSmith | Arize AI (Phoenix) |
| :---- | ----- | ----- | ----- |
| Supported Agent Frameworks | **Native Integrations:** LangChain, LlamaIndex, OpenAI SDK, LiteLLM, Dify, Flowise, Langflow.•   | **Native Integrations:** LangChain, LangGraph (first-party support). **Also Supported:** OpenAI SDK, LlamaIndex, AutoGen, CrewAI.  & any LLM workflow via the Python/JS SDK wrappers. | **Native Integrations:** LlamaIndex, LangChain, DSPy.•  Also Supported: OpenAI SDK, CrewAI, AutoGen, Mistral, Bedrock.• Standard:  Built on OpenInference, making it compatible with any framework that supports this standard. |
| Tracing Methods | **Decorator @observe()** (Python) automatically captures function inputs, outputs, and nesting. **OpenTelemetry:** Fully compatible backend; can ingest traces sent to /api/public/otel from any  **Auto-instrumentation:** Integrations (like langfuse-langchain) auto-capture traces without decorators. | **Decorator @traceable** wraps arbitrary functions to create trace runs. **Environment Variables:** Setting LANGCHAIN\_TRACING\_V2=true auto-traces all LangChain/LangGraph objects.  **Wrappers: wrap\_openai()** wraps the OpenAI client for auto-tracing. **OpenTelemetry:** Supports ingesting OTEL traces via langsmith\[otel\] and exporting to other OTEL collectors. | **OpenTelemetry:** Core architecture is OTEL-based; uses arize-phoenix-otel for instrumentation. **Auto-instrumentation:** register(auto\_instrument=True) automatically instruments supported libraries (OpenAI, LangChain, etc.) using OpenInference. **Decorators:** @tracer.chain and @tracer.tool for custom function tracing.  |
| **Evaluator Capabilities** |  **Model-Based Evaluation:** Managed "LLM-as-a-Judge" to score traces on criteria like hallucination, toxicity, and relevance. **Human Evaluation:** Annotation interface allows manual scoring and feedback on traces. **Dataset Experiments:** Run evaluations against versioned datasets; supports many-to-one mapping for regression testing.  **Libraries:** Integrates with Ragas for RAG-specific metrics (fidelity, answer relevance). | **Offline Evaluation:**  Code-based/ human eval with annotable queue and also code based eval.  Run chains over datasets using evaluate() to compute aggregate metrics (precision, recall). **Pairwise Comparison:** specialized views for A/B testing two different prompts/models side-by-side. **Online Evaluation:** Configurable evaluators that run automatically on sampled production traces.  **Unit Testing:** "Code evaluators" to validate structural correctness (e.g., valid JSON, regex matches). |  **Local Evaluation:**  The arize-phoenix-evals library runs evaluations locally on your machine (pandas-based) for fast iteration  **RAG Focus:** specialized evaluators for retrieval relevance (NDCG, Hit Rate, MRR) and embedding analysis. **Golden Datasets:** Tools to curate and export "golden" datasets from production traces for testing.•  Explanations: Evaluators generate text explanations for why a specific score was assigned. |

[image1]: <data:image/png;base64,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>
