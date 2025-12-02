# AMORE: Adaptive Multi-agent Orchestration with Reflective Execution

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)

> **AMORE** is a novel framework for adaptive multi-agent orchestration that dynamically selects collaboration patterns based on task complexity and execution feedback.

## Abstract

Large Language Model (LLM) based multi-agent systems have emerged as a promising paradigm for solving complex reasoning tasks. However, existing approaches suffer from three critical limitations: (1) **static orchestration** that cannot adapt to varying task complexity, (2) **error propagation** from early-stage failures in sequential pipelines, and (3) **inefficient resource allocation** treating all subtasks uniformly regardless of difficulty.

We introduce **AMORE** (Adaptive Multi-agent Orchestration with Reflective Execution), featuring three key innovations:

- **Complexity-Aware Router (CAR)**: Predicts optimal orchestration patterns (single-agent, parallel, hierarchical, or consensus) before execution
- **Reflective Checkpoint Mechanism (RCM)**: Enables mid-execution strategy adaptation through quality gates
- **Unified Memory Architecture (UMA)**: Three-tier memory system maintaining coherence across agent boundaries

## Key Results

| Benchmark | AMORE | Best Baseline | Improvement |
|-----------|-------|---------------|-------------|
| AgentBench | 59.7% | 53.9% | +10.8% |
| WebArena | 29.7% | 24.4% | +21.7% |
| MARS (Ours) | 52.3% | 44.1% | +18.6% |

**Cost Reduction**: 41% lower than AgentOrchestra through adaptive pattern selection.

## Repository Structure

```
AMORE/
├── experiments/
│   ├── amore_simulation.py       # Core AMORE simulation framework
│   ├── run_experiments.py        # Generate all paper tables
│   └── additional_experiments.py # Extended ablations and analysis
│
├── benchmark/
│   └── mars_specification.json   # MARS benchmark spec (2,847 tasks)
│
├── requirements.txt              # Python dependencies
├── LICENSE                       # Apache 2.0 License
└── README.md                     # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/muxiddin19/AMORE-ICML2026.git
cd AMORE-ICML2026

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running Experiments

### Generate All Paper Tables

```bash
cd experiments
python run_experiments.py
```

This generates:
- Tables 8-14 from the main paper
- Tables C1-C2 from the appendix
- `simulation_results.csv` with detailed results
- `paper_tables.json` for programmatic access

### Run Additional Ablations

```bash
python additional_experiments.py
```

This includes:
- CAR threshold sensitivity analysis
- RCM quality threshold experiments
- Cross-domain transfer evaluation
- Scalability stress tests
- Memory system ablations
- Budget constraint analysis
- Failure recovery analysis
- LLM backbone comparison

## MARS Benchmark

The Multi-Agent Reasoning Suite (MARS) benchmark includes 2,847 tasks across three domains:

| Category | Tasks | Avg. Subtasks | Complexity |
|----------|-------|---------------|------------|
| Scientific Research | 847 | 12.3 | Medium-High |
| Software Engineering | 1,124 | 8.7 | Low-High |
| Strategic Planning | 876 | 15.1 | High |

See `benchmark/mars_specification.json` for the full specification.

## Framework Components

### Complexity-Aware Router (CAR)

Extracts 7 features per subtask:
- `f_length`: Estimated reasoning steps
- `f_tools`: Tool diversity required
- `f_knowledge`: Knowledge domain breadth
- `f_ambiguity`: Task specification uncertainty
- `f_deps`: Upstream dependencies
- `f_critical`: Critical path position
- `f_history`: Historical success rates

### Orchestration Patterns

| Pattern | Agents | Cost | Latency | Best For |
|---------|--------|------|---------|----------|
| Single | 1 | 1× | 1× | Simple, clear tasks |
| Parallel | 2-3 | 2× | 1× | Data gathering, coverage |
| Hierarchical | 3-5 | 3-5× | 2× | Complex coordination |
| Consensus | 3+ | 5-10× | 3× | Critical decisions |

### Reflective Checkpoint Mechanism (RCM)

Quality gates with four actions:
- **Proceed**: Quality ≥ θ_high → Continue
- **Retry**: θ_low ≤ Quality < θ_high → Re-execute
- **Escalate**: Quality < θ_low → More powerful pattern
- **Replan**: Structural issue → Redecompose

### Unified Memory Architecture (UMA)

Three-tier cognitive architecture:
- **Working Memory**: Current context, per-agent
- **Episodic Memory**: Execution traces, session-level
- **Semantic Memory**: Consolidated knowledge, permanent

## Citation

```bibtex
@inproceedings{anonymous2026amore,
  title={AMORE: Adaptive Multi-agent Orchestration with Reflective Execution for Complex Reasoning Tasks},
  author={Anonymous},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about this research, please open an issue or contact the authors.

---

**Note**: This repository contains the code for the ICML 2026 submission.
