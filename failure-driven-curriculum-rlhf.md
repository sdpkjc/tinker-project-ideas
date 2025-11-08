# Failure-driven curriculum learning for RLHF: Training on what models get wrong

Standard RLHF trains uniformly on all prompts, but models have specific weaknesses. Failure-driven curriculum focuses training on examples the model currently fails, providing targeted improvement. This adapts curriculum learning principles ([Bengio et al., 2009](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/15972/Bengio%2C%202009%20Curriculum%20Learning.pdf)) to RLHF with automatic failure detection.

## Core idea

**Observation**: Models waste time on examples they already solve well

**Proposal**: Dynamically focus training on current failures

**Process**:
1. Evaluate model on task distribution
2. Identify failure modes and error patterns
3. Generate/select training examples targeting those failures
4. Train intensively on failures
5. Re-evaluate and iterate

**Benefits**:
- Efficient training (focus on weaknesses)
- Targeted improvement (fix specific errors)
- Automatic curriculum (naturally adapts)

## Failure identification

### Method 1: Accuracy-based

**Simple**: Track per-prompt success rate
- Low success → High priority for training
- High success → Lower priority

### Method 2: Error pattern clustering

**Advanced**: Cluster failures by type
- Factual errors
- Reasoning failures
- Format violations
- Safety issues

**Focus training on each cluster**

### Method 3: Reward model uncertainty

**Use RM**: High uncertainty → Likely failure region
- Train where RM is least confident
- Assumes uncertainty correlates with difficulty

### Method 4: Human feedback loops

**Explicit**: Ask users to flag failures
- Collect failure examples from deployment
- Train specifically on those

## Curriculum strategies

### Strategy 1: Hard example mining

**Method**:
1. Generate/sample many examples
2. Test current policy on all
3. Select hardest X% for training
4. Repeat

**Similar to**: Hard negative mining in computer vision

### Strategy 2: Error-specific data generation

**Method**:
1. Identify error type (e.g., "struggles with multi-digit multiplication")
2. Generate many examples of that type
3. Train until error rate drops
4. Move to next error type

### Strategy 3: Adaptive sampling

**Method**:
Dynamically adjust sampling distribution
- Higher probability for failure-prone prompts
- Lower probability for mastered prompts

**Implementation**: Prioritized experience replay

### Strategy 4: Hierarchical curriculum

**Method**:
1. Fix coarse-grained failures first
2. Then fine-grained failures
3. Hierarchy of difficulty

**Example**:
- Level 1: Basic instruction following
- Level 2: Multi-step instructions
- Level 3: Complex reasoning

## Training methodology

**Standard RLHF**: Uniform sampling

**Failure-driven**:
```
for each training iteration:
  current_failures = identify_failures(model)
  training_batch = sample_from_failures(current_failures)
  update_model(training_batch)
```

**Balancing**:
- Mix failure examples + general examples
- Avoid catastrophic forgetting on mastered examples

## Evaluation

### Metric 1: Sample efficiency

**Measure**: Performance vs. training examples
- Compare: Failure-driven vs. uniform sampling

**Expected**: Faster improvement with failure focus

### Metric 2: Error reduction

**Track**: Specific error types over training
- Does failure-driven reduce targeted errors faster?

### Metric 3: Generalization

**Test**: Does focusing on failures hurt generalization?
- Or improve by addressing weaknesses?

### Metric 4: Catastrophic forgetting

**Monitor**: Performance on mastered examples
- Ensure focusing on failures doesn't break existing capabilities

## Research questions

1. **Does failure-driven training improve efficiency?**
   - Faster learning? Better final performance?

2. **What failure identification method works best?**
   - Accuracy, clustering, uncertainty, human feedback?

3. **What's the optimal failure/general mix?**
   - 100% failures? 50/50 mix?

4. **Does this cause catastrophic forgetting?**
   - How to balance new learning vs. retention?

5. **Can we automate the full curriculum?**
   - Or requires human-in-the-loop?

## Applications

- **Rapid prototyping**: Quickly fix model weaknesses
- **Safety**: Focus on safety-critical failures
- **Domain adaptation**: Target domain-specific errors
- **User feedback integration**: Incorporate deployment failures

## Extensions

- **Multi-model curriculum**: Different models focus on different failures
- **Collaborative debugging**: Ensemble of models, train on individual failures
- **Meta-learning curriculum**: Learn curriculum strategy itself
- **Continual failure tracking**: Ongoing failure detection and mitigation in deployment
