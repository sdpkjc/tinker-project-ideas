# Uncertainty-aware calibration via self-consistency and introspection

Large language models are often confidently wrong. While self-consistency improves accuracy by sampling multiple reasoning traces, it also produces a powerful signal about uncertainty: when independent samples disagree, the model is likely wrong. This project turns that agreement pattern into a first-class calibration signal and teaches models to know when to be unsure, ask clarifying questions, or defer.

## Core idea

Use multiple diverse samples per prompt to estimate uncertainty, then train a lightweight calibrator (or a calibration head on the backbone) to predict correctness probability from features of the sample set and the model's own introspective signals. At inference time, allocate compute adaptively (more samples when needed) and support selective answering or deferral.

## Proposed approach

1. Diverse sampling:
   - Generate K reasoning traces per prompt (vary temperature, prompts, decoding);
   - Optionally align steps across traces to compute disagreement at the step level.
2. Agreement-derived features:
   - Majority vote margin, entropy over final answers, pairwise answer edit distance;
   - Trace-level features: mean length, variance in rationale length, step overlap;
   - Introspection: have the model verbalize confidence/assumptions; embed these.
3. Calibrator training:
   - Train a small classifier/regressor to predict P(correct) using the above features;
   - Compare post-hoc (logistic/temperature scaling) vs. joint training of a head.
4. Policy tuning for honesty:
   - Add a loss that aligns verbalized confidence with empirical correctness;
   - Penalize overconfidence; reward asking for clarification under high uncertainty.
5. Compute allocation:
   - Learn a policy to choose K adaptively given a budget and target risk level;
   - Stop early when confidence crosses a threshold (“anytime” self-consistency).

## Evaluation

- Benchmarks: math (GSM8K/MATH), code (HumanEval/MBPP), factual QA (TruthfulQA), instruction following (AlpacaEval/MT-Bench style judges);
- Metrics: ECE, Brier score, AURC (risk–coverage), selective accuracy, deferral utility;
- Ablations: impact of K, feature subsets, introspection prompts, base model size;
- Cost–quality tradeoffs: Pareto front of tokens vs. calibrated performance.

## Research questions

1. Which disagreement features carry the most calibration signal across domains?
2. Can introspective explanations improve calibration beyond distributional features?
3. How well does the calibrator transfer across tasks and base models?
4. Can we train the base model to be inherently better-calibrated (not just a head)?
5. What is the optimal policy for adaptive K under latency/quality constraints?

## Extensions

- Step-level PRM: use consistency of intermediate steps to teach process-level calibration;
- Deferral routing: when uncertain, defer to tools/humans/specialist models;
- Safety: treat high-uncertainty regions as likely unsafe; tighten guardrails there;
- Streaming decode: make token-by-token uncertainty available for interactive UIs.

