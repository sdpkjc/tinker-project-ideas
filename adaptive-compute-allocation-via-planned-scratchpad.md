# Adaptive compute allocation with planned scratchpads and early stopping

Chain-of-thought improves accuracy but increases latency and cost. Many prompts are easy and don’t need long reasoning, while hard ones benefit from deeper thinking. This project trains models to plan how much compute to spend—deciding when to use scratchpads, how long to think, and when to stop early without sacrificing reliability.

## Concept

Frame test-time compute as a decision problem. The model first emits a short plan about needed effort (“fast answer,” “brief rationale,” or “extended analysis”), then either answers directly or generates a bounded scratchpad. A controller monitors signals (confidence, contradiction, diminishing returns) to stop early or escalate.

## Approach

1. Planning tokens:
   - Add a lightweight preamble where the model predicts an effort level and rationale budget;
   - Supervise from difficulty proxies (agreement under self-consistency, retrieval scores, prior attempts).
2. Early stopping signals:
   - Monitor answer stability across samples, confidence heads, and PRM-style step quality;
   - Stop generation when stability is high; otherwise extend or switch strategy.
3. Cost-aware training:
   - RL with reward = task score – λ·tokens; vary λ to trace Pareto frontier;
   - Distill a small “compute policy” that maps prompt features to effort level.
4. Safety and deferral:
   - For safety-critical prompts, require higher certainty or explicit deferral;
   - Route unclear cases to tools or stronger models.

## Evaluation

- Tasks: math/code (clear quality signals), reasoning QA, multi-hop retrieval;
- Metrics: accuracy vs. tokens, response time, tail-latency percentiles, calibration under compute constraints;
- Baselines: always-cot, never-cot, fixed-length CoT, simple self-consistency.

## Research questions

1. Can models learn reliable “effort planning” tokens that generalize across domains?
2. Which signals best predict that more tokens will help (vs. waste compute)?
3. How to combine early stopping with selective answering and deferral policies?
4. Do compute policies transfer across model sizes or require per-model tuning?

## Extensions

- Mixture-of-compute experts: specialized small/medium/large heads selected by the planner;
- Streaming UIs: show partial thoughts only when helpful; hide for easy cases;
- Budget-aware batch scheduling for throughput-constrained deployments.

