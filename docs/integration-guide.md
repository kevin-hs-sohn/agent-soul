# Integration Guide

How to connect Agent Soul to your AI agent.

## Overview

Integration has three parts:
1. **Preprocessing** — convert your agent's session logs to training format
2. **Agent prompt** — tell your main LLM when and how to use soul_output
3. **Automation** — set up nightly training pipeline

## 1. Session Log Format

### Expected Input

The preprocessing script (`preprocess.py`) expects OpenClaw session JSONL format:

```jsonl
{"type": "message", "message": {"role": "user", "content": "서버 상태 확인해줘"}}
{"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "확인할게."}]}}
{"type": "message", "message": {"role": "toolResult", "content": "active (running)", "toolName": "exec"}}
```

### Adapting for Other Frameworks

If your agent doesn't use OpenClaw, you have two options:

**Option A: Write a custom parser.** Replace `parse_session_file()` and the content extraction functions in `preprocess.py`. Keep the rest (PII redaction, filtering, output format).

**Option B: Convert to intermediate format.** Write a separate script that converts your logs to the expected JSONL format, then use `preprocess.py` as-is.

### Output Format

Regardless of input format, the training data output is always:

```jsonl
{"text": "Kevin: 서버 상태 확인해줘\nAssistant: 확인할게.\n[exec result: active (running)]\nKevin: 고마워\nAssistant: ㅇㅇ"}
```

Simple clean text. One session per line. The model learns next-token prediction on this text.

## 2. Agent System Prompt

Add these sections to your agent's system prompt (AGENTS.md, system message, etc.):

### Soul Query Instructions

```markdown
## Soul Query (CRITICAL)

Before responding to the user, query the Soul — your past memory.
soul_output = a response generated purely from your past memories.
Treat it like recalling from your own memory — context, patterns, and
preferences are reliable, but factual claims (numbers, dates, prices)
may be misremembered. Verify facts when needed.

### How to Query
  Run: soul-query.sh "user message here"
  With context: soul-query.sh --context "recent conversation" "message"

### MUST Query When:
- User asks about past events, decisions, or preferences not in your loaded files
- User references something you have no context for
- First substantive message in a new session
- You need project history, patterns, or user's past stance on a topic
- Any memory-related question

### Skip When:
- Clear commands with full context given
- Mid-conversation follow-ups where context is already loaded
- Heartbeat/cron tasks
- Trivial file operations

### Default: Query.
When unsure whether to query → query.
The cost (~20-30s latency) is worth avoiding a context-less answer.

Don't show soul_output to user. Server unavailable → proceed without.
```

### Boot Sequence

Add soul query check to your agent's startup routine:

```markdown
## Every Session
1. Read identity files
2. Read memory/context files
3. Load working state
4. **Before each response: Check Soul Query rules — if conditions match, query soul BEFORE answering**
```

Making it a numbered procedural step (not just a guideline) is critical. LLMs follow step-by-step instructions more reliably than declarative rules buried in documentation.

## 3. Interpreting soul_output

The key framing that works well:

```
soul_output = a response generated from your past memories.
```

This framing works because:
- LLMs naturally understand "memory" as context-reliable but fact-unreliable
- It motivates the agent to use soul_output for context/patterns while being skeptical of specific claims
- It's more intuitive than "output from a small fine-tuned LLM"

### What to Trust

| Reliable | Unreliable |
|----------|------------|
| User preferences and patterns | Specific numbers, dates, prices |
| Project context and history | Exact commands or API details |
| Communication style | Technical specifications |
| Past decisions and rationale | Current system state |

## 4. Nightly Training Automation

See `examples/nightly-train.sh` for a reference implementation. The general pattern:

```
1. Collect today's session logs
2. Preprocess to training format
3. Start GPU instance (if using cloud GPU)
4. Train (incremental from previous adapter)
5. Convert to GGUF
6. Deploy to inference server
7. Health check (rollback if failed)
8. Stop GPU instance
```

### Cost Optimization

- Use spot/preemptible GPU instances (~$0.20-0.50/run for 15-30 min)
- Only train when new data exists (skip if no sessions today)
- Keep adapter count limited (delete old ones, keep last 5)
- Stop GPU instance immediately after training (trap cleanup on failure)

## 5. Inference Optimization

### System Prompt

Keep the Soul's system prompt minimal. The Soul's knowledge comes from LoRA weights, not from context injection. A heavy system prompt wastes tokens on CPU inference.

Good (fast):
```
/no_think
You are an assistant. Kevin is a developer in Seoul. Be concise.
Continue naturally based on patterns you've learned.
```

Bad (slow):
```
/no_think
[full SOUL.md, 3KB]
[full USER.md, 800B]
[full MEMORY.md, 700B]
[full SCRATCHPAD section, 2KB]
```

The bad version can cause timeouts on CPU inference. If you need recent context, inject only the last few lines of your scratchpad.

### Token Limits

For CPU inference, keep `max_tokens` low (128-192). Soul output doesn't need to be long — a few sentences of context is enough for the main LLM to work with. Longer output = longer wait.

## 6. Monitoring

Track these to know if Soul is working:

- **Response time**: Should be under 30s on CPU, under 5s on GPU
- **Empty responses**: If soul-query.sh returns empty frequently, check server health
- **Training loss**: Should generally decrease over time. Sudden spikes may indicate data quality issues
- **Qualitative check**: Periodically ask memory-related questions and compare soul_output quality
