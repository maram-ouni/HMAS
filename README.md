# HMAS — Supplementary Code

This archive contains the reference implementation of the HMAS (Hybrid
Multi-Agent System) architecture described in the accompanying paper,
across the two benchmarks used for evaluation.

## Structure

```
benchmarks/
  real/            REAL benchmark implementation (web-task environment)
  deepjsoneval/    DeepJSONEval benchmark implementation (structured JSON generation)
```

Both implementations follow the same conceptual architecture described in
the paper (Section 3): a `BDIPlannerAgent` running the belief-desire-intention
deliberation cycle (perceive -> brf -> reconsider -> options -> filter ->
execute), an `LLMExecutorAgent` that grounds each planned action via an LLM
call, and a `RuleBasedVerifierAgent` that scores each attempt as an
independent two-player verification game (Section 3.2), communicating over
a shared `MessageBus`.

The two benchmarks differ substantially in action space, observation
format, and evaluation criteria, so each was implemented as a self-contained
package rather than factored into a single shared library. Module names
mirror each other (`core/bdi_model.py`, `core/world_model.py`,
`agents/bdi_planner_agent.py`, `agents/rule_based_verifier.py`, etc.) to
make the architectural correspondence explicit, but the code itself is
benchmark-specific. Each benchmark also includes its own baseline
implementations (ReAct, Reflexion, ChatBDI, Hierarchical BDI, BDI-LLM Dual)
used for the comparisons reported in the paper, plus the ablation and
robustness-campaign scripts under `experiments/`.

## Setup

Each benchmark directory is a standalone project:

```
cd benchmarks/real          # or benchmarks/deepjsoneval
pip install -r requirements.txt
cp .env.example .env        # set your own LLM API key (e.g. GROQ_API_KEY)
```

No API keys or credentials are included in this archive; all secrets are
read from environment variables at runtime (see `config.py` in each
directory).

### Running the unit tests without an LLM key

Most of the test suite (166/166 for `real`, 69/69 for `deepjsoneval`) is
pure unit tests on the deterministic core logic (payoff computation, plan
selection, verifier calibration, etc.) and needs no network access. A
handful of modules instantiate an OpenAI-compatible client at import time,
so a non-empty placeholder key is enough to collect and run the full suite:

```
GROQ_API_KEY=dummy OPENAI_API_KEY=dummy python -m pytest tests/
```

## Notes for reviewers

- `benchmarks/real` corresponds to the architecture used for the main
  results table and the ablation/robustness experiments (Sections 5.1-5.3
  of the paper).
- `benchmarks/deepjsoneval` corresponds to the generalization experiments
  (Section 5.4).
- Author-identifying comments and local file paths have been removed from
  the source for double-blind review; this does not affect functionality.
