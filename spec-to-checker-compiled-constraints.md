# Compiling natural-language specs into executable checkers for alignment

Natural-language policies (constitutions, guidelines, specs) are flexible but ambiguous. This project explores translating those policies into executable validators—tests, schemas, and property checks—that can be embedded in training and evaluation. The goal is to make constraints explicit, auditable, and cheap to enforce during learning and inference.

## Motivation

- Current safety/quality objectives are often implicit in reward models;
- Hand-written filters are brittle, narrow, and hard to maintain;
- Many policies are inherently programmatic (format, citations, sources, safety checks) and could be auto-compiled into tests.

## Core approach

1. Spec parsing:
   - Take a policy written in natural language (e.g., “cite sources with URLs; avoid medical advice; answer concisely unless asked for detail”);
   - Use an LLM to propose a structured specification (YAML/JSON) of constraints.
2. Checker synthesis:
   - Generate executable validators: regexes, Pydantic schemas, Python functions, unit tests;
   - Include counterexample generators for adversarial fuzzing.
3. Round-trip validation:
   - From the checker, auto-generate examples that should pass/fail; verify with humans or a stronger model;
   - Measure precision/recall of the checker w.r.t. human judgment.
4. Integration into training:
   - Use compiled constraints as a multiplicative gate C(x, y) on reward (R = C·Q) or as hard filters during data collection;
   - Provide structured feedback to the policy (which constraint failed and why) for targeted fixes.

## Evaluation

- Domains: formatting tasks, citation-required QA, safety policies, chain-of-thought hygiene;
- Metrics: checker precision/recall, false-negative rate on critical constraints, reward hacking resistance, policy performance with and without compiled constraints;
- Robustness: adversarial paraphrase resistance; stability under minor spec changes.

## Research questions

1. How reliably can LLMs translate ambiguous policies into faithful checkers?
2. What verification loops (property-based testing, metamorphic tests) best raise trust?
3. Does factorizing constraints from quality (C·Q) reduce harmful failure modes?
4. Can we learn reusable constraint libraries and compose them safely?

## Extensions

- Spec diffing: explain behavior changes introduced by spec updates;
- Runtime guardrails: deploy checkers as lightweight, low-latency filters with actionable error messages;
- Cross-model portability: train checkers once, reuse across model families;
- Governance: maintain a versioned policy–checker registry with audit trails.

