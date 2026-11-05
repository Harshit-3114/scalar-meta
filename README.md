# Scalar Meta — Customer Support Ticket Triage

An LLM benchmark environment for routing customer support tickets to the correct department. Part of the [OpenEnv](https://github.com/huggingface/OpenEnv) ecosystem.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-FFD21E?logo=huggingface&logoColor=black)
![Tests](https://img.shields.io/badge/tests-25%20passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

---

## Overview

The agent receives a support ticket and must reply with one of four departments:

| Department | Example Ticket |
|---|---|
| `repair` | "My screen is cracked." |
| `billing` | "Charged twice!" |
| `sales` | "Want to buy more seats." |
| `tech` | "The app keeps crashing." |

### Tasks

Three difficulty levels:

| Task | Tickets | Description |
|---|---|---|
| `easy` | 1 | Single repair ticket |
| `medium` | 3 | Repair, billing, and sales |
| `hard` | 5 | Repair, billing, sales, tech, billing |

The final score is `correct_routing / total_tickets`.

The `hard` task is designed to trip naive classifiers: it contains two billing tickets with very different phrasings ("Charged twice!" vs. "I hate this, give me my money back immediately"), and a `tech` ticket ("The app keeps crashing on startup") that a keyword heuristic would likely misroute to `repair`.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key
export OPENAI_API_KEY=sk-...

# Run all tasks
python inference.py

# Run specific tasks with a custom model
python inference.py --tasks easy medium --model gpt-4o-mini --verbose
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY = "sk-..."
python inference.py
```

---

## CLI

```
usage: inference.py [-h] [--tasks [{easy,medium,hard} ...]]
                    [--model MODEL] [--api-base API_BASE]
                    [--api-key API_KEY] [--verbose]

Customer support ticket triage inference runner

options:
  -h, --help            show this help message and exit
  --tasks [{easy,medium,hard} ...]
                        Tasks to run (default: all)
  --model MODEL         Model name (default: gpt-3.5-turbo, or MODEL_NAME env var)
  --api-base API_BASE   API base URL (default: https://api.openai.com/v1)
  --api-key API_KEY     API key (default: OPENAI_API_KEY or HF_TOKEN env var)
  --verbose, -v         Enable debug logging
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `HF_TOKEN` | No | — | Fallback if `OPENAI_API_KEY` is unset |
| `MODEL_NAME` | No | `gpt-3.5-turbo` | Model to use |
| `API_BASE_URL` | No | `https://api.openai.com/v1` | API base URL |

Copy `.env.example` to `.env` and fill in your values (if using `python-dotenv`).

Because the runner accepts any OpenAI-compatible `--api-base`, you can point it at a local server (Ollama, vLLM, LM Studio) or any hosted provider without changing code.

---

## Environment API

The environment follows the standard Gym-style interface, which is what OpenEnv environments implement:

```python
from env import TicketTriageEnv

env = TicketTriageEnv(task="hard")
state = env.reset()                      # {"status": "in_progress", "pending_ticket": "..."}
state, reward, done, info = env.step("repair")
score = env.grader()                     # 0.0 – 1.0
```

| Method | Returns |
|---|---|
| `reset()` | Fresh state, tickets reset |
| `state()` | Current pending ticket or `{"status": "complete"}` |
| `step(action)` | `(state, reward, done, info)` — reward is `1/len(tickets)` per correct routing |
| `grader()` | Final score, `correct_routing / total_tickets` |

---

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v
```

25 unit tests cover task validation, reward arithmetic, ticket sequencing, the done-flag lifecycle, and the grader.

---

## Project Structure

```
scalar-meta/
├── env.py                  # TicketTriageEnv — Gym-style environment
├── inference.py            # CLI inference runner
├── tests/
│   ├── __init__.py
│   └── test_env.py         # Unit tests (25 tests)
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── openenv.yaml
├── Dockerfile
├── LICENSE
└── README.md
```

---

## Docker

```bash
docker build -t scalar-meta .
docker run --env OPENAI_API_KEY=sk-... scalar-meta
```

---

## Notes

- **Python 3.10+** is required — the type annotations use PEP 604 `X | Y` union syntax.
- The reward is deliberately fractional (`1/len(tickets)`) rather than binary, so partial credit is visible in the reward signal during multi-ticket tasks.
- Ticket order is fixed per task and deterministic, which makes runs directly comparable across models.

---

## License

MIT — see [LICENSE](LICENSE).
