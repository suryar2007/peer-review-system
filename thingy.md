# Build Guide: AI Peer Review Verification System

---

## Part 1 — External Setup (Do This Before Opening Cursor)

Complete all of these before writing any code. Each step produces a credential you'll need in your `.env` file.

---

### 1.1 — Python environment

You need Python 3.11+. Check with `python --version`. If you're on an older version, install via pyenv or python.org.

```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows
```

---

### 1.2 — Nous Research API (Hermes)

1. Go to **portal.nousresearch.com**
2. Create an account
3. Navigate to API Keys → Create new key
4. Copy the key — you only see it once
5. You get $5 free credits to start. Hermes 4 70B is $0.13/M input tokens so this goes a long way.

```
NOUS_API_KEY=your_key_here
NOUS_BASE_URL=https://inference-api.nousresearch.com/v1
```

Alternatively if the portal is down: go to **openrouter.ai**, create an account, and use `nousresearch/hermes-4-70b` — same models, slightly different pricing. Replace the base URL with `https://openrouter.ai/api/v1` and use your OpenRouter key.

---

### 1.3 — Lava MCP Tools

1. Go to **lava.so**
2. Sign up for a developer account
3. Navigate to the MCP catalog — you should see the Knowledge & Reference section with arXiv, Semantic Scholar, Wikipedia, Open Library, etc.
4. Get your API key from the dashboard
5. Note the MCP server URL for each tool you want to use — format is typically `https://mcp.lava.so/{tool-name}`

```
LAVA_API_KEY=your_key_here
```

If Lava MCP tools require individual tool enablement, enable: arXiv, Semantic Scholar, Wikipedia, Open Library, NASA.

---

### 1.4 — K2 Think V2

**Option A — HuggingFace Inference API (easiest to start)**
1. Go to **huggingface.co** and create an account
2. Get a token from Settings → Access Tokens
3. The model is at `LLM360/K2-Think-V2`
4. Enable the serverless inference API on the model page (may require Pro subscription)

```
HF_TOKEN=your_token_here
K2_MODEL_ID=LLM360/K2-Think-V2
```

**Option B — Self-hosted (recommended for production)**
You need a machine with 8× A100 80GB or equivalent. Run via vLLM:
```bash
pip install vllm
vllm serve LLM360/K2-Think-V2 \
  --tensor-parallel-size 8 \
  --max-model-len 131072 \
  --host 0.0.0.0 \
  --port 8000
```
Then set `K2_BASE_URL=http://localhost:8000/v1`

**Option C — Use a provider** (Fireworks AI or Together AI sometimes host the model):
Check fireworks.ai and together.ai for current availability.

---

### 1.5 — LangSmith (observability for LangGraph)

1. Go to **smith.langchain.com**
2. Create a free account
3. Create a new project — call it `peer-review-pipeline`
4. Get your API key from Settings

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key_here
LANGCHAIN_PROJECT=peer-review-pipeline
```

This gives you a full trace of every agent call, tool invocation, and state transition. Invaluable for debugging.

---

### 1.6 — Hex AI

1. Go to **hex.tech** and create an account (free Community tier works to start)
2. Create a new project — call it `Peer Review Dashboard`
3. Get your API token from Settings → API
4. Note your project ID from the URL: `app.hex.tech/workspace/.../projects/{PROJECT_ID}/...`

```
HEX_API_KEY=your_key_here
HEX_PROJECT_ID=your_project_id_here
```

You'll build the actual dashboard in Hex's UI in Part 3 of this guide. For now just get the credentials.

---

### 1.7 — E2B (sandboxed code execution for reproducibility checks)

1. Go to **e2b.dev**
2. Sign up and get an API key from the dashboard

```
E2B_API_KEY=your_key_here
```

---

### 1.8 — Semantic Scholar API key (direct fallback)

Lava wraps Semantic Scholar but having a direct key is useful as a fallback.

1. Go to **semanticscholar.org/product/api**
2. Request an API key (free, takes ~1 day)

```
SEMANTIC_SCHOLAR_API_KEY=your_key_here
```

---

## Part 2 — Build in Cursor

Open Cursor. Create a new folder `peer-review-system` and open it. Now work through these prompts in order. Each prompt builds on the previous one — don't skip ahead.

---

### Step 2.1 — Project Scaffold

**Cursor prompt:**

```
Create a Python project structure for an AI-powered academic peer review verification system. The system is a multi-agent pipeline that takes a PDF of an academic paper, extracts all citations and empirical claims, verifies citations against real academic databases, and uses a reasoning LLM to check whether cited sources actually support the claims attributed to them.

Create the following directory structure:

peer-review-system/
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
├── main.py
├── config.py
├── pipeline/
│   ├── __init__.py
│   ├── state.py          # LangGraph state schema
│   ├── graph.py          # LangGraph pipeline definition
│   └── nodes/
│       ├── __init__.py
│       ├── extractor.py       # Hermes extraction node
│       ├── citation_resolver.py  # Lava MCP retrieval node
│       ├── reasoner.py        # K2 Think V2 reasoning node
│       └── reporter.py        # Hex dashboard trigger node
├── agents/
│   ├── __init__.py
│   ├── hermes.py          # Hermes API client wrapper
│   ├── k2.py              # K2 Think V2 client wrapper
│   └── lava_tools.py      # Lava MCP tool wrappers
├── utils/
│   ├── __init__.py
│   ├── pdf_parser.py      # PDF text and structure extraction
│   └── hex_client.py      # Hex API client
└── tests/
    ├── __init__.py
    └── test_extraction.py

Populate requirements.txt with:
- langgraph>=0.2.0
- langchain>=0.3.0
- langchain-openai
- langsmith
- openai
- anthropic
- requests
- httpx
- pydantic>=2.0
- pymupdf  (for PDF parsing, imported as fitz)
- python-dotenv
- fastapi
- uvicorn
- networkx
- pandas

Populate .env.example with all environment variables needed:
NOUS_API_KEY=
NOUS_BASE_URL=https://inference-api.nousresearch.com/v1
HF_TOKEN=
K2_MODEL_ID=LLM360/K2-Think-V2
K2_BASE_URL=
LAVA_API_KEY=
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=peer-review-pipeline
HEX_API_KEY=
HEX_PROJECT_ID=
E2B_API_KEY=
SEMANTIC_SCHOLAR_API_KEY=

Create config.py that loads all env vars using python-dotenv and exposes them as typed constants with sensible error messages if required vars are missing.

Create a .gitignore that excludes .env, __pycache__, .venv, *.pyc, and any PDF test files.
```

---

### Step 2.2 — LangGraph State Schema

**Cursor prompt:**

```
In pipeline/state.py, define the complete LangGraph state schema for the peer review pipeline using Pydantic v2 and TypedDict.

The state needs to track everything that flows through the pipeline from paper ingestion to final dashboard. Define it as follows:

```python
from typing import TypedDict, Annotated, Optional
import operator
from pydantic import BaseModel

# First define the core data models as Pydantic classes:

class Citation(BaseModel):
    raw_text: str                    # The raw citation string from the paper
    authors: list[str]
    title: Optional[str]
    journal: Optional[str]
    year: Optional[int]
    doi: Optional[str]
    arxiv_id: Optional[str]
    url: Optional[str]
    resolved: bool = False           # Whether we found it in a database
    source_text: Optional[str] = None  # Retrieved abstract/full text
    exists: Optional[bool] = None    # Confirmed to exist in a real database

class Claim(BaseModel):
    text: str                        # The empirical claim made in the paper
    paper_section: str               # Which section it appears in
    supporting_citation_indices: list[int]  # Indices into the citations list
    claim_type: str                  # "empirical", "statistical", "methodological"

class StatisticalAssertion(BaseModel):
    text: str
    p_value: Optional[float]
    effect_size: Optional[float]
    sample_size: Optional[int]
    confidence_interval: Optional[tuple[float, float]]
    section: str

class VerificationResult(BaseModel):
    claim_text: str
    verdict: str                     # "supported", "overstated", "contradicted", "out_of_scope", "unverifiable"
    confidence: float                # 0.0 to 1.0
    explanation: str
    relevant_passage: Optional[str]  # The passage from the source that determined the verdict
    citation_indices: list[int]

class StatisticalAuditResult(BaseModel):
    assertion_text: str
    is_internally_consistent: bool
    issues: list[str]
    severity: str                    # "low", "medium", "high"

# Then define the pipeline state TypedDict:

class PipelineState(TypedDict):
    # Input
    paper_path: str
    paper_text: str
    paper_title: Optional[str]
    paper_abstract: Optional[str]
    
    # Phase 1 outputs (Hermes extraction)
    citations: list[Citation]
    claims: list[Claim]
    statistical_assertions: list[StatisticalAssertion]
    
    # Phase 2 outputs (Lava retrieval)
    resolved_citations: list[Citation]   # citations with source_text populated
    
    # Phase 3 outputs (K2 reasoning)
    verification_results: list[VerificationResult]
    statistical_audit_results: list[StatisticalAuditResult]
    
    # Phase 4 outputs (reporting)
    hex_run_id: Optional[str]
    dashboard_url: Optional[str]
    
    # Pipeline control
    current_phase: str
    errors: Annotated[list[str], operator.add]  # accumulates errors without overwriting
    
    # Summary stats (computed in reporter)
    total_citations: int
    resolved_count: int
    hallucinated_count: int
    supported_claims: int
    flagged_claims: int
```

Also create a helper function `initial_state(paper_path: str) -> PipelineState` that returns a state with all lists empty and defaults set.
```

---

### Step 2.3 — PDF Parser

**Cursor prompt:**

```
In utils/pdf_parser.py, build a robust PDF parser using PyMuPDF (fitz) that extracts structured content from academic papers.

The parser needs to:

1. Extract the full text with section boundaries preserved
2. Identify the References/Bibliography section specifically
3. Extract the paper title (usually the first large text on page 1)
4. Extract the abstract

Create a class PaperParser with these methods:

class PaperParser:
    def __init__(self, pdf_path: str):
        ...
    
    def parse(self) -> dict:
        # Returns: {
        #   "title": str,
        #   "abstract": str,
        #   "full_text": str,
        #   "sections": dict[str, str],  # section_name -> section_text
        #   "references_raw": str,        # raw references section text
        #   "page_count": int
        # }

For the references section: look for common heading patterns like "References", "Bibliography", "Works Cited". Extract everything after that heading to the end of the document. This is the raw text that Hermes will parse into structured Citation objects.

For sections: split on headings that match patterns like numbered headings (1., 2., 1.1) or all-caps headings or headings that appear in larger font size. Return a dict mapping section name to section text.

Handle these edge cases:
- Multi-column PDFs (merge columns left-to-right per page)
- Papers with no clear abstract heading (use the first paragraph after the title if it's >100 words)
- Very long reference lists (don't truncate — return all of it)

Add a simple __main__ block so it can be tested directly: python utils/pdf_parser.py path/to/paper.pdf and it prints the extracted title, abstract (first 200 chars), section names, and reference count (count newlines in references_raw as a rough proxy).
```

---

### Step 2.4 — Hermes Agent

**Cursor prompt:**

```
In agents/hermes.py, build a client wrapper for the Nous Research Hermes API.

Hermes uses an OpenAI-compatible API. Use the openai Python client pointed at the Nous base URL and API key from config.py.

Create a class HermesAgent with these methods:

class HermesAgent:
    def __init__(self):
        # Initialize openai client with NOUS_BASE_URL and NOUS_API_KEY
        # Default model: "hermes-4-70b" — check the Nous portal for the exact model string
        ...

    def extract_citations(self, references_raw: str) -> list[dict]:
        """
        Takes the raw references section text from a paper.
        Returns a list of structured citation dicts matching the Citation model.
        
        System prompt: You are a citation extraction specialist. Extract every reference 
        from the provided bibliography section. For each reference, extract all available 
        fields. Return ONLY a JSON array, no other text.
        
        Use JSON mode / response_format={"type": "json_object"} to force structured output.
        Ask it to return: {"citations": [...]} where each item has:
        authors (list), title, journal, year, doi, arxiv_id, url, raw_text
        """

    def extract_claims(self, paper_text: str, sections: dict) -> list[dict]:
        """
        Takes the full paper text and section dict.
        Returns a list of empirical claims, each with:
        text, paper_section, supporting_citation_indices (as [idx] where idx is 
        the position in the references list, parsed from in-text citation markers),
        claim_type (empirical/statistical/methodological)
        
        Focus on claims that are supported by specific citations (i.e. sentences 
        that end with [1], (Smith et al., 2023), etc.). Skip background statements 
        that have no citation. Limit to 50 most important claims.
        
        Return {"claims": [...]}
        """

    def extract_statistical_assertions(self, paper_text: str) -> list[dict]:
        """
        Extract every sentence that reports a statistical result.
        Look for: p-values (p=, p<, p>), effect sizes (d=, r=, β=), 
        confidence intervals ([x, y] or (x, y) near %), sample sizes (n=, N=).
        
        Return {"assertions": [...]} where each has:
        text, p_value (float or null), effect_size (float or null), 
        sample_size (int or null), confidence_interval ([low, high] or null), section
        """

Make each method handle API errors gracefully — retry once on rate limit (429), raise a descriptive exception on other errors. Log the token usage for each call.

Add a test at the bottom: if __name__ == "__main__": load a sample reference string and run extract_citations, print the result.
```

---

### Step 2.5 — Lava MCP Knowledge Tools

**Cursor prompt:**

```
In agents/lava_tools.py, build wrappers for Lava's MCP knowledge tools that resolve academic citations.

Lava provides MCP (Model Context Protocol) tools for arXiv, Semantic Scholar, Wikipedia, and Open Library. We need to call these to verify citations exist and retrieve their content.

Look up the current Lava MCP documentation at lava.so/docs to get the exact endpoint format and authentication method. The general pattern for MCP tool calls is HTTP POST to the MCP server URL with the tool name and arguments.

Build a class LavaKnowledgeTools with these methods:

class LavaKnowledgeTools:
    def __init__(self):
        # Load LAVA_API_KEY from config
        # Set base headers with authorization
        ...
    
    def search_arxiv(self, title: str, authors: list[str] = None, arxiv_id: str = None) -> dict | None:
        """
        Search arXiv for a paper by title or ID.
        Returns: {exists: bool, abstract: str, authors: list, year: int, url: str} or None
        Try arxiv_id first if provided (most reliable). Fall back to title search.
        """

    def search_semantic_scholar(self, title: str, doi: str = None, authors: list[str] = None) -> dict | None:
        """
        Search Semantic Scholar for a paper.
        Returns: {exists: bool, abstract: str, tldr: str, citation_count: int, year: int, url: str}
        Try DOI first if provided. Fall back to title search.
        """

    def search_wikipedia(self, query: str) -> dict | None:
        """
        Search Wikipedia for a topic.
        Returns: {exists: bool, summary: str, url: str}
        Used for cross-checking factual background claims.
        """
    
    def resolve_citation(self, citation: dict) -> dict:
        """
        Master method: try to resolve a citation using all available tools.
        Strategy:
        1. If arxiv_id exists, try search_arxiv first
        2. If doi exists, try search_semantic_scholar first  
        3. Otherwise try both with title, take whichever returns a result
        4. If nothing found, mark as unresolvable
        
        Returns the citation dict updated with:
        - resolved: True/False
        - exists: True/False  
        - source_text: the abstract or summary (used by K2 for reasoning)
        """

Also build a direct fallback method resolve_via_semantic_scholar_direct(title, doi) that calls the Semantic Scholar API directly (api.semanticscholar.org/graph/v1/paper/search) using SEMANTIC_SCHOLAR_API_KEY. Use this if the Lava tool fails.

Add rate limiting: sleep 0.5s between Lava calls to avoid hitting rate limits. Run resolutions concurrently using asyncio if the method is called in batch.
```

---

### Step 2.6 — K2 Think V2 Reasoning Agent

**Cursor prompt:**

```
In agents/k2.py, build the reasoning agent that uses K2 Think V2 (or a compatible reasoning model) to verify whether cited sources actually support the claims attributed to them.

K2 is accessed via an OpenAI-compatible API. Use NOUS_BASE_URL/K2_BASE_URL and the appropriate key from config.py. If K2_BASE_URL is set, use that (self-hosted); otherwise fall back to HuggingFace inference.

class K2ReasoningAgent:
    def __init__(self):
        # Initialize client, set model to K2_MODEL_ID
        # Set a high max_tokens (4096) to allow detailed reasoning
        ...

    def verify_claim(self, claim_text: str, cited_sources: list[dict], paper_context: str = "") -> dict:
        """
        Core verification method. 
        
        claim_text: the exact claim made in the paper
        cited_sources: list of {title, abstract/source_text} for each cited paper
        paper_context: the surrounding paragraph for context
        
        System prompt should instruct K2 to:
        - Read the claim carefully
        - Read each cited source
        - Determine: does the cited source actually support this claim?
        - Produce a verdict: "supported", "overstated", "contradicted", "out_of_scope", "unverifiable"
        - Explain the verdict in 2-3 sentences
        - Quote the most relevant passage from the source
        - Give a confidence score 0.0-1.0
        
        The prompt must emphasize: be skeptical. Do not give benefit of the doubt.
        If the source only tangentially relates to the claim, mark it "out_of_scope".
        If the source supports a weaker version of the claim, mark it "overstated".
        
        Return JSON: {verdict, confidence, explanation, relevant_passage}
        """

    def audit_statistics(self, assertions: list[dict]) -> list[dict]:
        """
        Check whether reported statistical values are internally consistent.
        
        For each assertion, check:
        - Is the p-value consistent with the reported sample size and effect size?
        - Are confidence intervals plausible given n?
        - Are there impossible values (p < 0, n < 0, CI where low > high)?
        - Does the precision of the p-value suggest rounding or reporting issues?
        
        Use a single prompt with all assertions batched (up to 20 at a time).
        
        Return list of {assertion_text, is_internally_consistent, issues: list[str], severity: "low"|"medium"|"high"}
        """

    def reconstruct_evidence_chain(self, main_conclusion: str, key_citations: list[dict]) -> dict:
        """
        For the paper's primary conclusion, trace the evidence chain.
        Does the evidence cited actually establish this conclusion,
        or does it only establish weaker intermediate results?
        
        Returns {chain_valid: bool, weakest_link: str, explanation: str}
        """

Important: wrap each call in a try/except. If K2 is unavailable, log the error and return a result with verdict="unverifiable" and a note that the reasoning model was unavailable. Never let a K2 failure crash the pipeline.
```

---

### Step 2.7 — Pipeline Nodes

**Cursor prompt:**

```
Build the four LangGraph pipeline nodes in pipeline/nodes/.

Each node is a function that takes PipelineState and returns a dict of state updates.

---

pipeline/nodes/extractor.py:

from pipeline.state import PipelineState
from agents.hermes import HermesAgent
from utils.pdf_parser import PaperParser

def extraction_node(state: PipelineState) -> dict:
    """
    Phase 1: Parse PDF and extract citations, claims, and statistical assertions.
    Uses PaperParser for PDF parsing, HermesAgent for structured extraction.
    
    1. Parse the PDF at state["paper_path"]
    2. Update state with paper_text, paper_title, paper_abstract
    3. Run Hermes extract_citations on the references section
    4. Run Hermes extract_claims on the full text
    5. Run Hermes extract_statistical_assertions on the full text
    6. Convert all dicts to Pydantic model instances
    7. Return state updates: citations, claims, statistical_assertions, paper_text, paper_title, paper_abstract
    
    On any error: append to state["errors"] and return whatever was successfully extracted.
    Always update current_phase to "extraction_complete".
    """

---

pipeline/nodes/citation_resolver.py:

from pipeline.state import PipelineState
from agents.lava_tools import LavaKnowledgeTools
import asyncio

def citation_resolution_node(state: PipelineState) -> dict:
    """
    Phase 2: Resolve all extracted citations against real databases.
    Uses LavaKnowledgeTools.resolve_citation for each citation.
    
    Run resolutions concurrently using asyncio.gather (wrap sync calls in executor).
    Limit concurrency to 5 at a time to avoid rate limits.
    
    Update each Citation object with: resolved, exists, source_text.
    Track counts: how many resolved, how many confirmed to exist, how many failed.
    
    Return: resolved_citations (updated list), total_citations count
    Always update current_phase to "resolution_complete".
    """

---

pipeline/nodes/reasoner.py:

from pipeline.state import PipelineState  
from agents.k2 import K2ReasoningAgent

def reasoning_node(state: PipelineState) -> dict:
    """
    Phase 3: Deep verification using K2 Think V2.
    
    For each claim in state["claims"]:
    - Look up its supporting citations from resolved_citations
    - Only verify claims where at least one citation was successfully resolved
    - Call K2ReasoningAgent.verify_claim with the claim and retrieved source texts
    - Build a VerificationResult
    
    Then batch all statistical_assertions into K2ReasoningAgent.audit_statistics.
    
    Run claim verification concurrently with a limit of 3 at a time (K2 is compute-heavy).
    
    Return: verification_results, statistical_audit_results
    Update summary counts: supported_claims, flagged_claims.
    Update current_phase to "reasoning_complete".
    """

---

pipeline/nodes/reporter.py:

from pipeline.state import PipelineState
from utils.hex_client import HexClient

def reporting_node(state: PipelineState) -> dict:
    """
    Phase 4: Package all results and trigger the Hex dashboard.
    
    1. Build a summary payload from the pipeline state:
       - All citations with their verification status
       - All claims with verdicts
       - All statistical audit results
       - Summary statistics
    
    2. Call HexClient.trigger_run(payload) to run the Hex notebook with this data
    3. Get back a run_id and poll until the run is complete (poll every 5s, timeout 120s)
    4. Get the dashboard URL from the completed run
    
    Return: hex_run_id, dashboard_url
    Update current_phase to "complete".
    Print a clear success message with the dashboard URL.
    """
```

---

### Step 2.8 — LangGraph Pipeline Assembly

**Cursor prompt:**

```
In pipeline/graph.py, assemble the complete LangGraph pipeline.

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pipeline.state import PipelineState, initial_state
from pipeline.nodes.extractor import extraction_node
from pipeline.nodes.citation_resolver import citation_resolution_node
from pipeline.nodes.reasoner import reasoning_node
from pipeline.nodes.reporter import reporting_node

Build the graph as follows:

1. Create a StateGraph with PipelineState as the schema

2. Add nodes:
   - "extract" → extraction_node
   - "resolve" → citation_resolution_node  
   - "reason" → reasoning_node
   - "report" → reporting_node

3. Add edges:
   - START → "extract"
   - "extract" → "resolve"
   - "resolve" → "reason"
   - "reason" → "report"
   - "report" → END

4. Add a conditional edge after "extract":
   If state["citations"] is empty (extraction failed completely), go to END with an error.
   Otherwise proceed to "resolve".

5. Compile with MemorySaver as the checkpointer so runs can be resumed if they fail.

6. Create a function run_pipeline(paper_path: str) -> PipelineState that:
   - Creates initial_state(paper_path)
   - Generates a unique thread_id (use uuid4)
   - Invokes the compiled graph with {"configurable": {"thread_id": thread_id}}
   - Returns the final state
   - Prints progress at each phase (use a callback or just print in each node)

7. Also create async_run_pipeline(paper_path: str) for async contexts.

Add clear docstrings explaining the pipeline flow.
```

---

### Step 2.9 — Hex Client

**Cursor prompt:**

```
In utils/hex_client.py, build a client for the Hex API that triggers notebook runs and retrieves dashboard URLs.

Hex has a REST API documented at learn.hex.tech/docs/api. The key endpoints are:
- POST /api/v1/project/{project_id}/run — trigger a run
- GET /api/v1/project/{project_id}/run/{run_id} — check run status

class HexClient:
    def __init__(self):
        # Load HEX_API_KEY and HEX_PROJECT_ID from config
        # Base URL: https://app.hex.tech
        # Auth header: Authorization: Token {HEX_API_KEY}
        ...
    
    def trigger_run(self, input_params: dict) -> str:
        """
        Trigger a Hex project run with the given input parameters.
        
        POST to /api/v1/project/{HEX_PROJECT_ID}/run
        Body: {
            "inputParams": {
                "paper_data": json.dumps(input_params)  
                # This is the variable name in your Hex notebook
            },
            "dryRun": false
        }
        
        Returns the run_id from the response.
        Raises HexError if the request fails.
        """
    
    def poll_run(self, run_id: str, timeout_seconds: int = 120) -> dict:
        """
        Poll a run until it completes or times out.
        
        GET /api/v1/project/{HEX_PROJECT_ID}/run/{run_id} every 5 seconds.
        Return the final run status dict when status is "COMPLETE" or "ERRORED".
        Raise TimeoutError if timeout_seconds is exceeded.
        """
    
    def get_dashboard_url(self, run_id: str) -> str:
        """
        Return the public URL for viewing the completed run's output.
        Format: https://app.hex.tech/app/{HEX_PROJECT_ID}?run={run_id}
        """
    
    def trigger_and_wait(self, input_params: dict) -> str:
        """
        Convenience: trigger a run, wait for completion, return dashboard URL.
        """

Add a mock mode: if HEX_PROJECT_ID is not set, return a dummy URL so the pipeline can be tested without a Hex account.
```

---

### Step 2.10 — Main Entry Point

**Cursor prompt:**

```
In main.py, build a clean CLI entry point for the pipeline.

import argparse
import sys
from pathlib import Path
from pipeline.graph import run_pipeline

Build a CLI with argparse that accepts:
  --paper PATH          Path to the PDF file (required)
  --output-json PATH    Optional: save the full pipeline state as JSON to this path
  --verbose             Print detailed logs during each phase
  --skip-reasoning      Skip K2 reasoning (faster, useful for testing extraction only)

The main function should:
1. Validate that the PDF path exists
2. Print a startup banner with the paper filename
3. Call run_pipeline(paper_path)
4. Print a formatted summary to stdout:
   - Paper title (if extracted)
   - Total citations found
   - Citations resolved: X/Y
   - Unresolvable citations: N (possible hallucinations)
   - Claims verified: N
   - Claims flagged: N (breakdown: overstated, contradicted, out_of_scope)
   - Statistical issues found: N
   - Dashboard URL (if Hex ran successfully)
5. If --output-json, serialize the final state and write to file
6. Exit with code 0 on success, 1 on complete failure

Also add a quick_test() function that runs the pipeline on a hardcoded test PDF for development. Document how to call it.
```

---

### Step 2.11 — Testing and Validation

**Cursor prompt:**

```
In tests/test_extraction.py, write tests for the extraction pipeline that don't require real API calls.

Use unittest.mock to patch the HermesAgent and LavaKnowledgeTools so tests run offline.

Write these tests:

1. test_pdf_parser_basic:
   - Create a minimal in-memory PDF using reportlab or just test with a real small PDF
   - Verify PaperParser returns a dict with all expected keys
   - Verify references_raw is non-empty

2. test_hermes_citation_extraction_schema:
   - Call HermesAgent.extract_citations with a sample references string (hardcode a 5-citation references section)
   - With the API mocked to return a valid JSON response
   - Verify all returned items have at minimum: title, authors, raw_text
   - Verify the Pydantic Citation model instantiates successfully from each result

3. test_citation_resolution_fallback:
   - Mock Lava tools to raise an exception
   - Verify the pipeline falls back to the direct Semantic Scholar API
   - Verify the citation is marked resolved=False rather than crashing

4. test_pipeline_state_accumulates_errors:
   - Set up a state where extraction partially fails
   - Verify errors list grows without overwriting previous errors
   - Verify the pipeline continues to the next phase despite errors

5. test_k2_unavailable_graceful_degradation:
   - Mock K2 to raise a ConnectionError
   - Verify verify_claim returns verdict="unverifiable" with an appropriate note
   - Verify the pipeline completes without crashing

Add a Makefile with:
- make install: pip install -r requirements.txt
- make test: python -m pytest tests/ -v
- make run PDF=path/to/paper.pdf: python main.py --paper $(PDF)
```

---

## Part 3 — Build the Hex Dashboard

Do this in the Hex web interface, not in Cursor.

---

### Step 3.1 — Create the Hex notebook

1. Open your Hex project at **app.hex.tech**
2. Create a new cell at the top — type: **Input** → **Text input**
   - Variable name: `paper_data`
   - Default value: `{}`
   - This is the JSON payload the API sends when it triggers a run

---

### Step 3.2 — Parse the input data

Add a Python cell:

```python
import json
import pandas as pd
import networkx as nx

# Parse the incoming pipeline data
data = json.loads(paper_data) if isinstance(paper_data, str) else paper_data

# Extract sub-datasets
citations = pd.DataFrame(data.get("citations", []))
claims = pd.DataFrame(data.get("verification_results", []))
stats = pd.DataFrame(data.get("statistical_audit_results", []))
summary = data.get("summary", {})

# Color encoding for citations
def citation_color(row):
    if not row.get("resolved"):
        return "red"      # unresolvable — possible hallucination
    if row.get("verdict") == "supported":
        return "green"
    if row.get("verdict") in ["overstated", "out_of_scope"]:
        return "amber"
    if row.get("verdict") == "contradicted":
        return "red"
    return "gray"         # resolved but not yet checked

if not citations.empty:
    citations["color"] = citations.apply(citation_color, axis=1)
    citations["status_label"] = citations["resolved"].map({True: "Resolved", False: "Not found"})
```

---

### Step 3.3 — Summary metric cards

Add a Python cell then a **Display** cell showing four metrics:

```python
total = summary.get("total_citations", len(citations))
resolved = summary.get("resolved_count", 0)
flagged_claims = summary.get("flagged_claims", 0)
stat_issues = len(stats[stats["severity"].isin(["medium", "high"])]) if not stats.empty else 0
```

Then add 4 **Metric** display cells referencing: `total`, `resolved`, `flagged_claims`, `stat_issues`

---

### Step 3.4 — Citation network graph

Add a Python cell:

```python
import plotly.graph_objects as go

# Build citation network using networkx
G = nx.DiGraph()

if not citations.empty:
    for i, row in citations.iterrows():
        G.add_node(i, 
                   label=row.get("title", row.get("raw_text", "Unknown"))[:50],
                   color=row.get("color", "gray"),
                   resolved=row.get("resolved", False))

    # Add edges from claims (paper → cited paper)
    for _, claim in claims.iterrows():
        for idx in claim.get("citation_indices", []):
            if idx in G.nodes:
                G.add_edge("paper", idx)

    pos = nx.spring_layout(G, seed=42)

    # Build plotly figure
    edge_x, edge_y = [], []
    for e in G.edges():
        x0, y0 = pos.get(e[0], (0,0))
        x1, y1 = pos.get(e[1], (0,0))
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_colors = [G.nodes[n].get("color", "gray") for n in G.nodes()]
    node_labels = [G.nodes[n].get("label", str(n)) for n in G.nodes()]

    color_map = {"green": "#1D9E75", "amber": "#EF9F27", "red": "#E24B4A", "gray": "#888780"}
    node_colors_hex = [color_map.get(c, "#888780") for c in node_colors]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=0.5, color="#ccc"), hoverinfo="none"))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                             marker=dict(size=12, color=node_colors_hex),
                             text=node_labels, textposition="top center",
                             hoverinfo="text"))
    fig.update_layout(showlegend=False, 
                      title="Citation network — green: supported, amber: overstated, red: flagged/not found",
                      height=500,
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)")

citation_graph = fig
```

Add a **Chart** display cell referencing `citation_graph`.

---

### Step 3.5 — Flagged claims table

Add a Python cell:

```python
if not claims.empty:
    flagged = claims[claims["verdict"] != "supported"].copy()
    flagged = flagged.sort_values("confidence", ascending=False)
    flagged_display = flagged[["claim_text", "verdict", "confidence", "explanation"]].copy()
    flagged_display.columns = ["Claim", "Verdict", "Confidence", "Explanation"]
    flagged_display["Confidence"] = flagged_display["Confidence"].apply(lambda x: f"{x:.0%}")
```

Add a **Table** display cell referencing `flagged_display`.

---

### Step 3.6 — Statistical audit table

Add a Python cell:

```python
if not stats.empty:
    stats_display = stats[["assertion_text", "severity", "issues", "is_internally_consistent"]].copy()
    stats_display["issues"] = stats_display["issues"].apply(lambda x: "; ".join(x) if isinstance(x, list) else x)
    stats_display.columns = ["Statistical claim", "Severity", "Issues found", "Consistent?"]
    stats_high = stats_display[stats_display["Severity"] == "high"]
```

Add a **Table** display cell referencing `stats_display`.

---

### Step 3.7 — Publish the notebook as a data app

1. Click **Publish** in the top-right of Hex
2. Under **App Settings**, set it to be publicly accessible (or share-by-link)
3. Note the app URL format — this is what gets returned to the reviewer

---

## Part 4 — End to End Test

### Step 4.1 — Find a test paper

Download a real preprint from arXiv that you can verify by hand. Good options:
- Any paper with a clean references section
- Even better: a paper that has known issues (Retraction Watch is a good source)

Save it as `tests/sample_paper.pdf`

---

### Step 4.2 — Run the pipeline

```bash
# Make sure your .env is populated
cp .env.example .env
# Fill in all your API keys

# Install dependencies
pip install -r requirements.txt

# Test PDF parsing first (no API calls)
python utils/pdf_parser.py tests/sample_paper.pdf

# Test Hermes extraction (makes API calls)
python agents/hermes.py

# Test Lava resolution (makes API calls)
python agents/lava_tools.py

# Run the full pipeline
python main.py --paper tests/sample_paper.pdf --verbose --output-json output.json
```

---

### Step 4.3 — Final Cursor prompt to fix integration issues

After running end-to-end for the first time, paste your error output into Cursor with this prompt:

```
The peer review pipeline ran end-to-end and produced the following errors. 
Fix each issue while preserving the overall architecture:

[paste your error output here]

Specific things to check:
1. Make sure the JSON schema from Hermes matches what the Pydantic Citation and Claim models expect
2. Make sure the Lava MCP tool response format is parsed correctly
3. Make sure the K2 prompt produces parseable JSON (add explicit JSON formatting instructions if needed)
4. Make sure the Hex trigger payload uses the correct variable name that matches the Hex notebook input cell
5. Make sure concurrent execution in citation_resolver doesn't cause race conditions on the state dict
```

---

## Reference: Full .env Template

```bash
# Hermes (Nous Research)
NOUS_API_KEY=
NOUS_BASE_URL=https://inference-api.nousresearch.com/v1

# K2 Think V2
HF_TOKEN=
K2_MODEL_ID=LLM360/K2-Think-V2
K2_BASE_URL=                        # Leave empty to use HF inference

# Lava knowledge tools
LAVA_API_KEY=

# LangSmith observability
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=peer-review-pipeline

# Hex dashboards
HEX_API_KEY=
HEX_PROJECT_ID=

# E2B sandboxed execution
E2B_API_KEY=

# Semantic Scholar (direct fallback)
SEMANTIC_SCHOLAR_API_KEY=
```

---

## What to Demo

For a 90-second demo at YHack:

1. Pick a paper from arXiv with at least 20 references
2. Run `python main.py --paper paper.pdf`
3. Show the terminal output — total citations, how many resolved, how many flagged
4. Open the Hex dashboard URL in a browser
5. Click a red node in the citation graph — show that it leads to a claim where the source doesn't support what the paper claims
6. Show the statistical audit table sorted by severity

The punchline: "In 45 seconds, this pipeline traced every citation in the paper back to its source and verified whether the evidence actually supports the claims. A human reviewer doing this manually would take 3-4 hours."
