# Soul Design — World Prediction Model

## Philosophy

**Soul = a world model built from agent-world interactions.**

Soul is not a memory store. It doesn't recall "this happened on Tuesday." It predicts "given this situation, the world will respond like this." The learning target is causal structure:

```
Given (context, action_t) → world_response_t+1
```

When the agent acts, the world responds. The pattern of these causal relationships accumulates in LoRA weights.

## Soul vs. Self Model

Soul does not model "what kind of entity am I." It models "when I act this way, the world responds that way."

| Self Model | World Model (Soul) |
|---|---|
| "I have this personality" | "This user reacts this way in this situation" |
| "I value these things" | "This system responds to this command like this" |
| Internal reflection | External prediction |

## Two Layers of Learning

### Intra-session (causal chains within a session)

- Agent runs code → error → fix → success
- Agent explains → user confused → re-explain → understanding
- Short causal chains naturally present in agent session traces

### Cross-session (recursive learning across sessions)

- Soul generates soul_output → influences agent behavior → world responds → becomes training data
- If Soul's prediction was useful → that trace gets trained on → similar predictions strengthen
- If not useful → prediction loss is high → naturally weakens
- Result: useful patterns recursively self-reinforce

## Training Data Structure

### Role Mapping

| Session Event | Training Role | Rationale |
|---|---|---|
| Assistant (text + tool calls + reasoning) | `human` (= action) | Agent's actions |
| User message | `gpt` (= world response) | World's reaction (user) |
| Tool result | `gpt` (= world response) | World's reaction (system) |

The model learns to predict world responses given agent actions.

### Preprocessing Pipeline

1. **Parse session logs** — skip metadata, extract message events
2. **Serialize content** — structured text, truncate long tool results (500 chars)
3. **Filter noise** — remove heartbeat/cron messages, skip short sessions
4. **PII redaction** — phone, email, API keys, wallet addresses, card numbers, mnemonics, IPs
5. **Output** — clean text JSONL: `{"text": "User: ...\nAssistant: ...\n[tool result: ...]"}`

## Architecture

```
User Message
    |
    v
Soul (LoRA small LLM)
    |  Predicts world response patterns
    |  Generates soul_output as context
    v
Main LLM (Claude, GPT-4, etc.)
    |  Receives soul_output + user message
    |  Reasons, decides, responds
    |  Optionally queries RAG for facts
    v
Response
```

**Key**: RAG and Soul are independent. The main LLM decides when to use RAG based on context + soul_output. Soul doesn't know RAG exists.

## Why Not Just RAG?

| | RAG | Soul |
|---|---|---|
| **What it stores** | Documents, facts | Behavioral patterns |
| **How it retrieves** | Keyword/embedding search | Generative prediction |
| **What it knows** | What's explicitly written | Implicit patterns never documented |
| **Latency** | Fast (search) | Slower (generation) |
| **Use case** | "What's the server IP?" | "What does the user prefer when debugging?" |

Both are useful. They complement each other.

## Technical Choices

| Choice | Value | Rationale |
|---|---|---|
| Base model | Qwen3-8B | Strong multilingual, good at pattern learning |
| Quantization | Q4_K_M (GGUF) | Fits in ~5GB RAM for CPU inference |
| LoRA rank | 16 | Enough capacity for personality/pattern learning |
| Training | QLoRA via Unsloth | 4-bit base + LoRA = low VRAM (~8GB) |
| Sequence length | 2048-4096 | Enough context per training example |
| Inference | llama.cpp CPU | No GPU needed for serving |

## Open Questions

### Thinking Content in Training
Agent's internal reasoning is included in training data. Whether this helps or hurts world prediction quality needs experimentation.

### Catastrophic Forgetting
Daily incremental LoRA training may lose early patterns over time. Possible mitigations:
- Periodic full retraining on all accumulated data
- "Constitution data" — curated examples that always get included
- Monitor prediction quality on reference prompts

### Cold Start
Base model has no world knowledge initially. The first few weeks of training data strongly influence long-term direction. Starting with data from a similar agent (warm start) can help.

### Optimal Training Frequency
Daily works well in practice. More frequent (hourly) adds marginal value at higher cost. Less frequent (weekly) risks stale patterns.
