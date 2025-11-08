# Few-shot reward model adaptation for new domains and tasks

Training reward models requires thousands of human preference labels, which is expensive and time-consuming. When deploying RLHF to new domains (medical, legal, scientific) or new tasks, collecting sufficient preference data for a new reward model is often a bottleneck.

This project explores few-shot reward model adaptation: quickly adapting existing reward models to new domains or tasks using only tens or hundreds of preferences, rather than thousands, inspired by few-shot learning ([Brown et al., 2020](https://arxiv.org/abs/2005.14165)) and meta-learning approaches.

## Problem statement

**Current RLHF workflow**:
1. Collect 10K-100K+ preference labels for target domain
2. Train reward model from scratch (or from pre-trained LM)
3. Use RM for RLHF in that domain

**Problem**: High data requirements for each new domain/task

**Desired**:
1. Start with general-purpose RM trained on broad data
2. Adapt to new domain with 50-500 preference labels
3. Achieve comparable performance to fully-trained domain-specific RM

**Why this matters**:
- Enables RLHF for niche domains (not enough data for full training)
- Faster iteration (prototype domain-specific RLHF quickly)
- Lower cost (fewer human annotations needed)
- Accessibility (smaller organizations can do domain-specific RLHF)

## Key challenges

**Challenge 1**: Catastrophic forgetting
- Fine-tuning on small domain-specific data may damage general capabilities

**Challenge 2**: Overfitting
- Small dataset risks overfitting to specific examples
- May not generalize to diverse domain examples

**Challenge 3**: Domain shift
- New domain may have different quality criteria
- General RM's learned features may not transfer

**Challenge 4**: Preference sparsity
- With only 50-500 labels, covering the domain is hard
- May miss important subcategories or edge cases

## Proposed approaches

### Approach 1: Parameter-efficient fine-tuning

Use adapter methods to adapt RM without modifying full model:

**Methods**:
- **LoRA** ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)): Low-rank adaptation matrices
- **Prefix tuning**: Learn soft prompts
- **Adapter layers**: Small bottleneck layers between transformer blocks

**Benefits**:
- Most parameters frozen → less forgetting
- Few trainable parameters → less overfitting
- Can maintain multiple domain adapters and switch between them

**Training**:
1. Freeze base RM parameters
2. Add small adapter (LoRA, prefix, etc.)
3. Train adapter on few-shot domain preferences
4. At inference: Base RM + domain adapter

### Approach 2: Meta-learning for fast adaptation

Train RM to be good at few-shot adaptation:

**Method** (MAML-style):
1. During meta-training:
   - Sample domain D from diverse set of domains
   - Sample small support set from D (50-500 examples)
   - Adapt RM to D using support set
   - Evaluate on query set from D
   - Optimize RM for fast adaptability

2. During deployment:
   - New domain arrives with small dataset
   - Adapt meta-trained RM (few gradient steps)
   - RM quickly specializes to new domain

**Benefits**:
- RM learns to extract domain-specific features quickly
- Principled approach to few-shot learning
- Can adapt to truly novel domains

### Approach 3: Mixture-of-experts with domain routing

Maintain general RM plus domain-specific experts:

**Architecture**:
```
Input → Domain classifier → Route to expert(s)
        ↓
        General RM (always active)
        Domain Expert 1
        Domain Expert 2
        ...
        → Weighted combination → Final score
```

**Training**:
1. Train general RM on broad data
2. For each new domain, train small expert on domain data
3. Train routing function to select relevant experts

**Benefits**:
- Modular: Add new domains without retraining everything
- Preserves general capability (general RM always contributes)
- Can blend multiple domains (weighted combination)

### Approach 4: Contrastive domain adaptation

Adapt RM using contrastive learning between general and domain-specific examples:

**Method**:
1. Start with general RM trained on broad data
2. Few-shot domain adaptation:
   - Positive examples: Domain-specific preferences
   - Negative examples: General examples that violate domain norms
   - Contrastive loss: Pull domain examples toward domain-appropriate rewards

**Benefits**:
- Explicitly learns what's different about new domain
- Can use unlabeled domain data (with pseudo-labels)
- Less prone to catastrophic forgetting

### Approach 5: Prompt-based domain conditioning

Condition RM on domain descriptions or examples:

**Method**:
```
Input: [Domain description] + [Prompt] + [Response] → RM score
```

**Domain description** can be:
- Text description: "This is a medical Q&A task. Prioritize accuracy and safety."
- Few-shot examples: 5-10 example preferences from the domain
- Domain embedding: Learned vector representing domain

**Training**:
1. Train RM with domain conditioning on diverse domains
2. At deployment, provide domain description for new domain
3. RM adapts behavior based on conditioning

**Benefits**:
- No parameter updates needed
- Instant adaptation to new domains
- Can interpolate between domains

## Data strategies for few-shot adaptation

**Active learning**:
- Intelligently select which examples to label
- Prioritize diverse, representative, or uncertain examples
- Maximize information from limited labels

**Data augmentation**:
- Generate synthetic preferences using prompted LLM
- Mix real few-shot labels with synthetic labels
- Increases effective dataset size

**Transfer from related domains**:
- Identify similar domains with existing data
- Initialize from related domain RM, adapt to new domain
- Leverages domain similarity

**Curriculum learning**:
- Start with general preferences, gradually introduce domain-specific
- Smooth transition prevents catastrophic forgetting

## Evaluation methodology

### Experiment 1: Few-shot adaptation effectiveness

**Setup**:
1. Hold out a domain (e.g., medical Q&A)
2. Train general RM on all other domains
3. Sample K domain-specific preferences (K = 10, 50, 100, 500)
4. Adapt RM using proposed methods
5. Evaluate on held-out domain test set

**Baselines**:
- No adaptation (general RM only)
- Full fine-tuning (may overfit or forget)
- From-scratch training (data-hungry, upper bound)

**Metrics**:
- Accuracy on domain preferences
- Ranking correlation with human judgments
- Policy quality after RL with adapted RM

### Experiment 2: Forgetting and generalization

**Question**: Does domain adaptation hurt general capabilities?

**Test**:
- Before adaptation: Evaluate RM on general test set
- After adaptation: Re-evaluate on same general test set
- Measure: Performance drop (forgetting)

**Also test**: Does adapted RM generalize to unseen domain examples?

### Experiment 3: Domain diversity

**Question**: How does adaptation differ across domain types?

**Test across**:
- Similar domains (general Q&A → scientific Q&A)
- Different domains (Q&A → creative writing)
- Specialized domains (general → medical, legal)

**Hypothesis**: Adaptation is easier for similar domains

### Experiment 4: Data efficiency

**Question**: How many examples are needed for effective adaptation?

**Test**: Vary K from 10 to 1000, measure adaptation quality

**Find**: Minimum viable dataset for each adaptation method

## Practical implementation

**Implementation in Tinker Cookbook**:
- Add few-shot adaptation module
- Support LoRA, meta-learning, and prompt-based conditioning
- Provide easy API for domain adaptation

**Workflow**:
1. Train general RM on diverse data
2. For new domain:
   - Collect 50-500 preference labels
   - Run few-shot adaptation script
   - Evaluate adapted RM
   - Deploy for RLHF in new domain

## Key research questions

1. **Can we adapt RMs with <100 examples?**
   - What's the minimum viable dataset?
   - How does this compare to fully-trained domain RM?

2. **Which adaptation method works best?**
   - LoRA, meta-learning, mixture-of-experts, prompting?
   - Trade-offs in performance, efficiency, and generalization?

3. **Does adaptation hurt general capabilities?**
   - Can we adapt without catastrophic forgetting?
   - Preserve general capabilities while specializing?

4. **How domain-specific are reward functions?**
   - Do different domains need very different RMs?
   - Or is adaptation mostly about calibration/emphasis?

5. **Can we predict adaptation difficulty?**
   - Which domains are easy vs. hard to adapt to?
   - Features that predict adaptation success?

## Expected outcomes

**If successful**:
- RLHF becomes accessible for niche domains
- Faster prototyping and iteration
- Lower cost and effort for domain-specific RLHF
- Understanding of domain transfer in reward modeling

**If partially successful**:
- Even with more data needed than hoped, adaptation is cheaper than from-scratch training
- Insights into what makes domains different
- Methods applicable to other transfer learning scenarios

## Extensions

**Multi-domain adaptation**:
- Adapt to multiple related domains simultaneously
- Share information across domains

**Continual domain adaptation**:
- Sequentially adapt to new domains over time
- Maintain performance on all previously-seen domains

**User-specific adaptation**:
- Adapt to individual user preferences (extreme few-shot)
- Personalization with minimal user feedback

**Cross-lingual adaptation**:
- Adapt English RM to other languages with few examples
- Leverage multilingual model capabilities

**Compositional domain adaptation**:
- New domain is combination of seen domains (e.g., "medical + creative writing")
- Compose adapters or interpolate between domain RMs

By enabling few-shot adaptation of reward models, we can make RLHF practical for the long tail of specialized domains and tasks, democratizing access to advanced post-training techniques beyond organizations with massive annotation budgets.
