# Recursive self-improvement in a sandboxed capability domain

Recursive self-improvement—where a model generates data to train itself iteratively—has been explored in limited forms ([STaR](https://arxiv.org/abs/2203.14465), [Self-Taught Reasoner](https://arxiv.org/abs/2212.10560)), but these approaches typically use external verification or conservative sampling to prevent capability collapse. What if we deliberately push recursive self-improvement to its limits in a *sandboxed domain* where we can safely study both success and failure modes?

## Motivation

Current self-improvement methods are conservative by necessity—they filter generated data heavily and often mix in human data. But to understand the fundamental dynamics of recursive self-improvement, we need to study the "pure" form where a model trains exclusively on its own outputs across multiple generations. The risk of capability collapse makes this dangerous in open-ended domains, but a carefully designed sandbox can make it tractable.

## Proposed sandbox domain: Formal logic proofs

Choose a domain with:
- Automatic verifiability (correct proofs can be mechanically checked)
- Clear capability boundaries (problems have measurable difficulty)
- Limited memorization risk (can generate infinite novel problems)
- Interpretable failure modes (invalid proofs are analyzable)

Example: Proving theorems in propositional logic or simple type theory.

## Experimental protocol

**Generation 0**: Start with a small seed dataset of human-written proofs (e.g., 100 examples)

**Generation N → N+1**:
1. Train model on all data from generation N
2. Generate a large corpus of candidate proofs for new problems
3. Filter only valid proofs (use automated verification)
4. Add all valid proofs to training set for generation N+1
5. No mixing of human data after generation 0

**Measurements at each generation**:
- Problem difficulty the model can solve (measured by proof search depth, axiom complexity)
- Proof efficiency (length, elegance metrics)
- Success rate on held-out test problems
- Diversity of proof strategies
- Signs of capability collapse or mode collapse

## Key research questions

1. **Does recursive self-improvement plateau or grow?**
   - Can models improve beyond human-level in the seed data?
   - Or do they hit a ceiling due to limited diversity in self-generated data?

2. **What causes capability collapse?**
   - If capability degrades over generations, can we characterize the mechanism?
   - Is it due to mode collapse, distributional shift, or accumulated bias?

3. **Can we predict and prevent collapse?**
   - Are there early warning signs (e.g., decreasing proof diversity)?
   - Do interventions like diversity bonuses, temperature annealing, or periodic resets help?

4. **How does model scale affect dynamics?**
   - Do larger models show more stable recursive improvement?
   - Is there a minimum scale required for positive feedback loops?

5. **Data efficiency vs. generations**:
   - Does generation N model learn faster than generation 0 from the same amount of data?
   - Can models learn to be better data generators over time?

## Extensions beyond the sandbox

If recursive self-improvement proves stable in formal domains:
- Test on verifiable but less constrained domains (code with unit tests)
- Gradually relax verification (use model-based judges instead of mechanical verification)
- Study transfer: does a model trained recursively generalize better to out-of-domain tasks?
- Explore "curriculum emergence": do models naturally generate data of increasing difficulty?

## Connection to AI safety

This project directly addresses questions about recursive self-improvement that are central to long-term AI safety discussions, but grounds them in empirical study rather than speculation. Understanding the conditions under which recursive self-improvement succeeds or fails is crucial for predicting the behavior of more capable future systems.

The sandbox approach lets us "run the experiment" on recursive self-improvement with full observability and safety, potentially revealing principles that apply more broadly.
