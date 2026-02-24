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

## Why "Soul"

Humans don't remember by writing sticky notes in their brain. Memory is encoded implicitly — distributed across neural network weights as patterns, associations, and intuitions. You can't point to a single neuron and say "that's where Tuesday's meeting is stored." It's embedded across the network.

Agent Soul works the same way. Past conversations aren't stored as retrievable documents (that's what RAG does). They're encoded into LoRA weights through fine-tuning — implicit, distributed, and generative. The agent doesn't "look up" a memory. It *recalls* — generating context from patterns baked into the model, just like you recall a colleague's communication style without consulting a file about them.

### Sleep and Memory Consolidation

Humans process the day's experiences during sleep. The brain replays events, strengthens useful patterns, and integrates new experiences into long-term memory. Without sleep, memories don't consolidate — they fade.

Agent Soul's nightly training is the same process. Each day's conversations are replayed through the model, patterns are reinforced in the LoRA weights, and the agent wakes up the next day with a richer, more consolidated memory. Skip the nightly training, and yesterday's interactions fade into nothing.

### Memory + Reasoning = Identity

The human brain separates memory and reasoning into distinct regions. The hippocampus handles memory formation and recall. The prefrontal cortex handles reasoning, planning, and decision-making. Neither alone makes a person — it's their combination that creates a coherent identity.

Current AI agents are prefrontal cortex only — pure reasoning, no memory. They're brilliant but amnesiac. Every conversation starts from zero. Agent Soul adds the missing piece: a memory module that carries implicit knowledge from past experiences. Together, the reasoning LLM and the Soul form something closer to a complete cognitive agent — one that *knows you*, not just one that *can think*.

The sum of an entity's memories is what makes it who it is. That's the soul.

### What Implicit Memory Can Do That RAG Can't

RAG is explicit memory — searchable, precise, but limited to what's been written down. Implicit memory captures things that were never documented:

| Scenario | RAG | Soul |
|----------|-----|------|
| "Fix this the way I like" | No document describes "how you like it" | Learned your debugging style from past sessions |
| User sends "k" or "yep" | No entry for this | Knows it means casual approval from past patterns |
| Predicting user frustration | Can't search for emotions | Recognizes patterns that preceded frustration before |
| Project context nobody wrote down | If it's not in a doc, it's gone | Absorbed from weeks of conversations about the project |
| "That thing we discussed last week" | Needs exact keywords to search | May recall the context and topic from trained patterns |

Explicit memory tells you what someone *said*. Implicit memory tells you who someone *is*.

## What You Get

After a few weeks of daily fine-tuning:

```
User: "What kind of person am I? Don't just list facts — analyze me."

Soul output: "You're a developer based in Seoul, working at the intersection
of crypto and AI. Your days are mostly code, system design, and model tuning.
You systematize your own growth through docs like AGENTS.md. Pragmatic,
results-oriented, always optimizing."
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
