# Peer review verification system

Multi-agent pipeline that ingests an academic PDF, extracts citations and empirical claims, resolves citations against academic databases (via Lava), and uses a reasoning model (K2 Think V2) to assess whether cited sources support the claims attributed to them. Results can be pushed to Hex for dashboards.

## Layout

- `pipeline/` — LangGraph `PipelineState`, compiled graph, and per-stage nodes (extract → resolve → reason → report).
- `agents/` — Thin clients for Hermes (Nous), K2, and Lava-shaped retrieval.
- `utils/` — PDF parsing (PyMuPDF) and Hex API helper.
- `config.py` — Environment-backed settings with validation.
- `main.py` — Command-line runner.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your keys
```

Required environment variables (see `config.py` and `.env.example`):

- `NOUS_API_KEY`, `LAVA_API_KEY`
- If `LANGCHAIN_TRACING_V2` is true: `LANGCHAIN_API_KEY`

Optional: `HEX_*` for reporting, `K2_BASE_URL` / `HF_TOKEN` for hosted K2, `SEMANTIC_SCHOLAR_API_KEY` for direct API use if you add it in retrieval code.

## Run

From the repository root:

```bash
python main.py path/to/paper.pdf
python main.py path/to/paper.pdf --json
```

## Tests

```bash
pytest tests/
```

## Notes

Hermes, Lava, and Hex integrations use placeholder URLs or behaviors where the public contract is not fixed in this scaffold—replace with your actual endpoints and MCP wiring.
