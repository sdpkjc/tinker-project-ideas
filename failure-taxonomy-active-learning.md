# Discovering and fixing failures via taxonomy-driven active learning

LLM failures are diverse: hallucination, sycophancy, unsafe advice, off-topic drift, formatting errors, brittle tool use, etc. Treating them as a single error category misses opportunities for targeted improvement. This project builds an evolving taxonomy of failures and uses it to drive data collection, training, and evaluation.

## High-level idea

Create a human-guided, model-assisted loop that (1) clusters failures into interpretable categories, (2) prioritizes undercovered or high-severity categories, and (3) generates counterexamples and targeted fixes. The taxonomy becomes a living map of model weaknesses and training progress.

## System components

1. Failure mining:
   - Collect failures from evals, red teaming, and user interactions;
   - Use embeddings + LLM summarization to propose cluster labels and exemplars.
2. Taxonomy curation:
   - Humans confirm/merge/split categories; assign severity, detectability, and prevalence;
   - Maintain a versioned taxonomy and a small rubric per category.
3. Active selection:
   - Choose prompts near category decision boundaries (uncertain classification);
   - Upweight rare but severe categories; propose perturbations for near-miss cases.
4. Counterexample generation:
   - Train/generate adversarial prompts that trigger each category;
   - Synthesize minimally edited positives to teach safe/accurate alternatives.
5. Targeted training:
   - Fine-tune or RL with category-aware sampling; add constraint-specific rewards;
   - Track per-category learning curves and regressions over time.

## Evaluation

- Metrics: coverage (share of failures taxonomized), discovery rate of new categories, fix rate per category, recurrence rate after fixes, precision of automated category classifier;
- Benchmarks: safety suites, knowledge QA, formatting-heavy tasks, tool-use tasks;
- Ablations: taxonomy depth, active learning policy, human effort vs. gains.

## Research questions

1. What representations best support interpretable, stable failure clusters?
2. Can we auto-detect when a category is “solved” or has morphed into subtypes?
3. How do category-specific fixes transfer across domains and models?
4. What is the optimal mix of human curation vs. model-driven discovery?

## Extensions

- Public leaderboards by failure category; 
- Cross-model generalization: do different families share failure topology?
- Tooling: lightweight UI to browse taxonomy, label examples, and launch fixes.

