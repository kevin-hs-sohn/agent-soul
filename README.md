# Agent Soul

Give your AI agent long-term memory through LoRA fine-tuning on its own conversation history.

Agent Soul is a small LoRA-tuned language model that runs alongside your main AI agent (Claude, GPT-4, etc.). It learns patterns from past conversations and provides context that the main model can't access — user preferences, project history, recurring patterns, and implicit knowledge built up over weeks of interaction.

## How It Works

```
User Message
    |
    v
+---------------------------+
|  Soul (LoRA, small LLM)   |  <-- Fine-tuned on past conversations
|  Generates soul_output:   |
|  context from "memory"    |
+------------+--------------+
             |
             v
+---------------------------+
|  Main LLM (Claude, etc.)  |  <-- Receives soul_output as context
|  Reasons + responds       |
+---------------------------+
             |
             v
        Response
```

**Soul is not RAG.** RAG retrieves specific facts from documents. Soul learns *patterns* — how your user communicates, what they care about, how past projects went, what approaches worked. It's the difference between searching your notes and actually remembering.

## What You Get

After a few weeks of daily fine-tuning:

```
User: "나는 어떤 사람이야? 분석해줘"

Soul output: "Kevin은 서울의 개발자야. 암호화폐와 AI 프로젝트를 다루는
전문가야. 코드 작성, 시스템 설계, AI 모델 개선이 주류야. AGENT.md 같은
문서로 자기 계발을 체계화하고 있어."
```

The main LLM receives this as context and can give a much richer, more personalized response than it could from scratch.

## Quick Start

### Prerequisites

- Python 3.10+
- A GPU for training (RTX 3090+ recommended, ~30 min/day)
- CPU server for inference (llama.cpp, ~5GB RAM)
- An AI agent that logs its conversations (any format — adapt `preprocess.py`)

### 1. Preprocess Session Logs

Convert your agent's conversation logs to training format:

```bash
python training/preprocess.py --input /path/to/session/logs --output train.jsonl
```

Input: JSONL session logs (see [docs/integration-guide.md](docs/integration-guide.md) for format)
Output: `{"text": "User: ...\nAssistant: ..."}` per session

### 2. Train the LoRA Adapter

```bash
pip install -r training/requirements.txt
python training/train.py \
  --data train.jsonl \
  --output ./output \
  --base-model unsloth/Qwen3-8B \
  --epochs 1 \
  --rank 16
```

For incremental training (recommended for daily updates):
```bash
python training/train.py \
  --data train.jsonl \
  --base-adapter ./previous_adapter \
  --output ./output
```

### 3. Convert to GGUF

```bash
bash training/convert_adapter.sh ./output/adapter ./output/soul-adapter.gguf
```

### 4. Run Inference Server

```bash
bash inference/soul-server.sh
# Starts llama-server on localhost:18791
```

### 5. Query the Soul

```bash
bash inference/soul-query.sh "What does the user usually want when they say 'check the server'?"
```

### 6. Integrate with Your Agent

Add to your agent's system prompt (see [docs/integration-guide.md](docs/integration-guide.md)):

```markdown
## Soul Query
Before responding, query your past memory:
  system.run ["/path/to/soul-query.sh", "user message here"]

soul_output = a response generated from your past memories.
Context and patterns are reliable; factual claims may be inaccurate.
```

## Architecture

```
training/
  preprocess.py       # Session logs → clean text JSONL
  train.py            # QLoRA fine-tuning (Unsloth + SFTTrainer)
  convert_adapter.sh  # PEFT safetensors → GGUF for llama.cpp
  requirements.txt    # Python dependencies

inference/
  soul-server.sh      # llama-server wrapper with Soul config
  soul-query.sh       # Query client (curl → llama-server API)
  soul-prompt-template.txt
  soul-inference.service  # systemd unit example

docs/
  design.md           # Architecture and training details
  integration-guide.md # How to integrate with your agent

examples/
  agent-prompt.md     # Example system prompt additions
  nightly-train.sh    # Example nightly automation script
```

## Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Base model | Qwen3-8B | Good multilingual support, efficient with QLoRA |
| Training method | QLoRA (rank 16) | Low VRAM, fast training, incremental-friendly |
| Inference | llama.cpp (CPU) | Runs on any server, no GPU needed |
| Adapter format | GGUF | Native llama.cpp support, single file |
| Training frequency | Nightly | Balances freshness vs. compute cost |
| Context to Soul | Minimal | Lightweight system prompt for fast CPU inference |

## Limitations

- **Cold start**: Soul knows nothing until it has a few sessions of training data. First useful results after ~20-30 sessions.
- **CPU inference is slow**: ~7 tok/s on 8-core CPU. Response in ~20-30s. GPU inference recommended for production.
- **Base model artifacts**: Occasional character mixing (e.g., Cyrillic in Korean text) from Qwen3's multilingual tokenizer.
- **Not a fact store**: Soul learns patterns and context, not precise facts. Always verify numbers, dates, and prices from soul_output.
- **Catastrophic forgetting**: Long-running incremental training may lose early patterns. Monitor and consider periodic full retraining.

## Adapting for Your Setup

The preprocessing script (`preprocess.py`) currently parses OpenClaw session JSONL format. To adapt for your agent framework:

1. Implement a parser that converts your session logs to the same output format: `{"text": "User: ...\nAssistant: ..."}` JSONL
2. Keep the PII redaction — it's framework-agnostic
3. Everything else (training, inference, integration) works as-is

See [docs/integration-guide.md](docs/integration-guide.md) for details on the expected format and how to write a custom parser.

## License

MIT
