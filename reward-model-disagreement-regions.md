# Identifying and characterizing reward model disagreement regions

In RLHF, we often train multiple reward models (different architectures, random seeds, or training data) to ensure robustness. These reward models frequently disagree on how to score certain responses. Understanding where and why disagreement occurs can reveal important insights about reward modeling limitations and guide improvements.

This project systematically studies reward model disagreement: identifying regions of high disagreement, characterizing what makes them challenging, and using disagreement as a signal to improve reward modeling and RL training.

## Motivation

**Observation**: Multiple reward models trained on the same data often disagree significantly on some examples.

**Questions**:
- What kinds of responses cause disagreement?
- Does disagreement indicate:
  - Genuinely ambiguous quality (both models uncertain)?
  - Modeling blind spots (neither model understands)?
  - Noise in training data (conflicting human preferences)?
  - Reward hacking opportunities (RL can exploit model-specific biases)?

**Hypothesis**: Disagreement regions reveal important information about:
- Reward model reliability and calibration
- Human preference ambiguity
- Potential reward hacking attack surfaces

## Research questions

1. **Where do reward models disagree?**
   - What fraction of responses have high RM disagreement?
   - Which domains/task types show most disagreement?
   - How does disagreement correlate with response properties?

2. **Why do reward models disagree?**
   - Ambiguous cases (both humans and RMs uncertain)?
   - Complex trade-offs (accuracy vs. helpfulness)?
   - Out-of-distribution (rare response types)?
   - Modeling artifacts (architectural biases)?

3. **Is disagreement predictable?**
   - Can we predict high-disagreement regions in advance?
   - What features correlate with RM disagreement?

4. **Does RL exploit disagreement regions?**
   - During RL training, do policies gravitate toward high-disagreement areas?
   - Is this reward hacking (exploiting model-specific biases)?

5. **Can we use disagreement to improve RLHF?**
   - Train better RMs by focusing on disagreement regions?
   - Avoid overoptimizing disagreement regions during RL?
   - Request human feedback specifically for high-disagreement cases?

## Experimental framework

### Setup: Train ensemble of reward models

**Method**:
1. Train N reward models (N = 5-10) with variation:
   - Different random seeds
   - Different architectures (small, medium, large)
   - Different training data subsets
   - Different training hyperparameters

2. For each (prompt, response), collect all N reward scores

3. Compute disagreement metrics:
   - **Variance**: σ²(R₁, R₂, ..., Rₙ)
   - **Range**: max(R_i) - min(R_i)
   - **Pairwise disagreement**: Fraction of pairs that disagree on ranking
   - **Entropy**: Treating normalized scores as probabilities

### Analysis 1: Characterizing disagreement regions

**Questions**:
- What percentage of responses have high disagreement (e.g., top 10% variance)?
- How does disagreement vary across:
  - Domains (math, code, creative writing)
  - Response length
  - Response quality (human-rated)
  - Prompt ambiguity

**Method**:
- Collect diverse (prompt, response) pairs
- Measure RM disagreement on each
- Correlate disagreement with response features

**Visualization**:
- Scatter plot: Response quality vs. RM disagreement
- Heatmap: Domain vs. disagreement level
- Histogram: Distribution of disagreement across dataset

### Analysis 2: Causes of disagreement

For high-disagreement examples, investigate why RMs disagree:

**Human annotation study**:
1. Sample 500 high-disagreement examples
2. Have humans judge:
   - Is this genuinely ambiguous or is there a clear best response?
   - What makes this hard to judge?
   - Do humans also disagree on this example?

**Feature analysis**:
- Train classifier to predict high vs. low disagreement
- Feature importance: Which features predict disagreement?
- Features to test:
  - Response length and complexity
  - Lexical diversity
  - Domain/topic
  - Formatting and structure
  - Presence of hedging language ("maybe", "possibly")

**Failure mode analysis**:
- Do different RMs exhibit systematic biases?
  - Length bias (some prefer longer, others shorter)
  - Format bias (some prefer bullet points)
  - Tone bias (some prefer formal, others casual)

### Analysis 3: Disagreement dynamics during RL

**Question**: Does RL training exploit disagreement regions?

**Method**:
1. Train policy with RL using one reward model (R₁)
2. Throughout training, measure disagreement:
   - Sample responses from policy at each checkpoint
   - Score with ensemble of RMs
   - Track disagreement over training

**Hypothesis**: If policy learns to exploit R₁, we expect:
- Increasing disagreement over training (policy moves toward R₁-specific high-reward regions)
- R₁ scores increase, but other RMs' scores plateau or decrease

**Visualization**: Plot training trajectory in (R₁ score, RM disagreement) space

### Analysis 4: Human alignment in disagreement regions

**Question**: When RMs disagree, which (if any) aligns with human preferences?

**Method**:
1. Sample high-disagreement response pairs
2. Collect human preference judgments
3. Compare:
   - Human consensus vs. each RM's prediction
   - Which RM best predicts human preferences in disagreement regions?

**Question**: Are disagreement regions genuinely ambiguous for humans too?

**Measure**: Human annotator agreement vs. RM ensemble agreement

### Analysis 5: Uncertainty estimation

**Question**: Do RMs "know" when they're in disagreement regions?

**Method**:
1. Train RMs with uncertainty estimation (e.g., ensembling, Bayesian methods)
2. Compare: Model uncertainty vs. ensemble disagreement
3. Test: Does high model uncertainty correlate with high ensemble disagreement?

**Use case**: If model uncertainty predicts disagreement, single RM can flag uncertain judgments without needing ensemble.

## Using disagreement to improve RLHF

### Application 1: Disagreement-aware RL training

Modify RL objective to account for disagreement:

**Conservative optimization**:
```
R_conservative(x, y) = mean(R_i) - λ · disagreement(R_i)
```

Penalize high-disagreement regions, avoid overoptimizing uncertain areas.

**Consensus requirement**:
Only optimize when ensemble agrees (low disagreement).

**Robust optimization**:
Optimize worst-case score across ensemble in high-disagreement regions.

### Application 2: Active learning for preference collection

Use disagreement to guide human feedback collection:

**Strategy**:
1. Generate responses from policy
2. Score with RM ensemble
3. Request human preferences specifically for high-disagreement examples
4. Retrain RMs with targeted feedback

**Benefit**: Efficiently allocate expensive human feedback to most informative examples.

### Application 3: RM training curriculum

**Idea**: Train RMs specifically on previously-disagreed examples

**Method**:
1. Identify high-disagreement examples from RM ensemble
2. Collect human ground truth for these examples
3. Fine-tune RMs on disagreement-rich dataset
4. Test: Does this reduce disagreement and improve accuracy?

### Application 4: Disagreement as exploration signal

In RL training, use disagreement for exploration:

**Method**:
- High disagreement → Policy uncertain about quality
- Add exploration bonus for discovering consensus-good, previously-disagreed examples
- Encourages policy to explore undermodeled regions

## Key research questions

1. **How common is RM disagreement?**
   - What fraction of examples show significant disagreement?

2. **What causes disagreement?**
   - Ambiguity, complexity, out-of-distribution, or modeling artifacts?

3. **Is disagreement a problem or a feature?**
   - Problem: Indicates RM unreliability
   - Feature: Reveals interesting edge cases and multi-modal preferences

4. **Does RL exploit disagreement?**
   - Do policies learn to hack model-specific biases?
   - Can we prevent this with ensemble RMs or disagreement penalties?

5. **Can disagreement improve RLHF?**
   - Better active learning
   - More robust RL training
   - Improved RM training

## Practical implementation

**Tools needed**:
- Efficient RM ensemble evaluation
- Disagreement metrics and visualization
- Integration with Tinker Cookbook RLHF pipeline

**Workflow**:
1. Train RM ensemble
2. Evaluate on diverse response dataset
3. Identify and analyze disagreement regions
4. Use findings to improve RM training and RL

## Expected outcomes

**Empirical findings**:
- Taxonomy of disagreement types
- Quantification of disagreement frequency
- Correlation between disagreement and other metrics

**Practical improvements**:
- Better active learning for preference collection
- More robust RL training procedures
- Guidelines for when to trust single RM vs. require ensemble

**Theoretical insights**:
- Understanding of RM limitations and failure modes
- Characterization of human preference ambiguity
- Connection between disagreement and reward hacking

## Extensions

**Disagreement across objectives**:
- Separate RMs for helpfulness, harmlessness, honesty
- Study disagreement between objectives (trade-offs)

**Temporal disagreement**:
- How does disagreement change as RMs train?
- Does disagreement decrease (models converge) or increase (models diverge)?

**Disagreement in continual learning**:
- Track disagreement as RMs are updated with new data
- Detect when new data introduces conflicting preferences

**Cross-model family disagreement**:
- Train RMs on different base models (Llama, GPT, Claude)
- Study architectural biases through disagreement patterns

By systematically studying where and why reward models disagree, we can build more robust RLHF systems that account for uncertainty, avoid exploitation of model-specific biases, and efficiently target human feedback collection.
