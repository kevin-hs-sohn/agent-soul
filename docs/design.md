# Design

## Core Idea

Agent Soul gives your AI agent persistent memory by fine-tuning a small LLM on its conversation history. The main LLM treats soul_output as "recalling from its own past memories" — context and patterns are reliable, factual claims may be inaccurate.

## Soul vs. RAG

| | RAG | Soul |
|---|---|---|
| **What it stores** | Documents, facts | Behavioral patterns, implicit knowledge |
| **How it retrieves** | Keyword/embedding search | Generative recall |
| **What it knows** | What's explicitly written | Things never documented — preferences, habits, communication style |
| **Latency** | Fast (search) | Slower (generation) |
| **Use case** | "What's the server IP?" | "What does the user prefer when debugging?" |

Both are useful. They complement each other. Soul doesn't replace RAG — it fills the gap RAG can't cover.

## How Training Works

### Role Mapping

Session logs are converted to training data with this role mapping:

| Session Event | Training Role | Rationale |
|---|---|---|
| Assistant (text + tool calls) | input | Agent's actions |
| User message | target | How the user responded |
| Tool result | target | How the system responded |

The model learns conversational patterns — how the user communicates, what they ask about, how they react to different approaches.

### Preprocessing Pipeline

1. **Parse session logs** — extract message events, skip metadata
2. **Serialize content** — clean text, truncate long tool results (500 chars)
3. **Filter noise** — remove heartbeat/cron messages, skip short sessions (<2 turns)
4. **PII redaction** — phone, email, API keys, wallet addresses, card numbers, mnemonics, IPs
5. **Output** — `{"text": "User: ...\nAssistant: ...\n[tool result: ...]"}` JSONL

### Incremental Training

Daily LoRA fine-tuning on new sessions, continuing from the previous adapter. This accumulates knowledge over time without full retraining.

Potential issue: catastrophic forgetting over many iterations. Mitigations:
- Periodic full retraining on all accumulated data
- Monitor recall quality on reference prompts
- Keep all preprocessed training data for retraining

## Architecture

```
User Message
    |
    v
Soul (LoRA small LLM)
    |  Recalls patterns from past conversations
    |  Generates soul_output as memory context
    v
Main LLM (Claude, GPT-4, etc.)
    |  Receives soul_output + user message
    |  Reasons, decides, responds
    |  Optionally queries RAG for facts
    v
Response
```

Soul and RAG are independent. The main LLM decides when to use RAG. Soul doesn't know RAG exists.

## Technical Choices

| Choice | Value | Why |
|---|---|---|
| Base model | Qwen3-8B | Strong multilingual, efficient with QLoRA |
| Quantization | Q4_K_M (GGUF) | ~5GB RAM for CPU inference |
| LoRA rank | 16 | Enough capacity for pattern learning |
| Training | QLoRA via Unsloth | Low VRAM (~8GB), fast, incremental-friendly |
| Inference | llama.cpp CPU | No GPU needed for serving |
| System prompt | Minimal (~200 bytes) | Soul's knowledge is in LoRA weights, not context. Heavy prompts waste CPU time. |

## Open Questions

### Cold Start
Base model has no conversation history. First useful recall after ~20-30 sessions of training data. Bootstrapping with data from a similar agent can help.

### Training Frequency
Daily works well. More frequent adds marginal value at higher cost.

### Base Model Artifacts
Qwen3's multilingual tokenizer occasionally produces character mixing (e.g., Cyrillic in Korean). This is a base model issue, not a LoRA issue.
