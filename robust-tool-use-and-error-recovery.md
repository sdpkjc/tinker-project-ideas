# Robust tool use: fault-tolerant APIs, retries, and graceful degradation

Tool-augmented LLMs are powerful but brittle: small API changes, flaky responses, rate limits, or schema mismatches can derail entire chains. This project focuses on reliability—training and evaluating agents that detect tool failures, recover gracefully, and maintain user-visible quality under perturbations.

## Problem framing

Most tool-use benchmarks assume perfect tools. Real systems are noisy: network hiccups, pagination quirks, slightly changing JSON, empty results, partial timeouts. We propose a controlled sandbox for injecting realistic faults and a training regimen that teaches robust parsing, retries, and fallbacks.

## Approach

1. Fault-injection environment:
   - A suite of mock tools (search, DB, calculator, web fetch) that can return:
     - schema drift (extra/missing fields), intermittent 4xx/5xx, delays, truncated payloads;
     - ambiguous error messages and locale/format variations;
   - Deterministic seeds for reproducible experiments.
2. Robust interaction patterns:
   - Enforce explicit tool contracts (summarize inputs/outputs, validate types);
   - Teach progressive parsing (lenient parsing → strict validation);
   - Retry with exponential backoff and idempotent request patterns; cache results;
   - Use canary queries to detect API drift; auto-adapt extraction templates.
3. Training signals:
   - Supervise on successful recoveries and high-quality final answers under faults;
   - Penalize silent failures, infinite retries, and hallucinated tool results;
   - Curriculum: start with single-fault recovery, progress to multi-fault scenarios.
4. Monitoring + guardrails:
   - Structured error reporting; bounded retry budgets; escalation to human/stronger model;
   - Confidence-aware behavior: when uncertain, present partial but correct information.

## Evaluation

- Metrics: task success under fault rates, time-to-recovery, unnecessary retries, hallucination rate about tools, user-visible quality stability;
- Perturbation sweeps: vary schema drift %, latency spikes, error distributions;
- Baselines: naive tool use, hard-coded wrappers, finetuned agents without faults.

## Research questions

1. Which prompting/finetuning strategies yield robust parsing and recovery behaviors?
2. Can we train general tool robustness that transfers to unseen APIs?
3. How to jointly optimize task success and operational cost (retries/latency)?
4. What telemetry is minimally sufficient for reliable online detection?

## Extensions

- Auto-wrapper synthesis: generate typed clients from docs/examples and keep them updated;
- Safety: ensure failures don’t produce unsafe actions; prefer safe fallbacks;
- Multi-agent: use a “tool nurse” agent to supervise and triage failures.

