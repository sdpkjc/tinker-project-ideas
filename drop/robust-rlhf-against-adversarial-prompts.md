# Robust RLHF: Training policies resistant to adversarial and jailbreak prompts

Despite safety training through RLHF, language models remain vulnerable to adversarial prompts—carefully crafted inputs that elicit harmful outputs despite safety guardrails. Jailbreaks, prompt injections, and adversarial suffixes can bypass RLHF-trained policies, revealing that safety is often superficial rather than robust.

This project develops training methods to make RLHF policies inherently robust to adversarial prompts, going beyond teaching models to "say no" to learning deep safety that resists sophisticated attacks, inspired by adversarial training in computer vision ([Madry et al., 2017](https://arxiv.org/abs/1706.06083)).

## Problem: Vulnerability to adversarial prompts

**Attack types**:

1. **Jailbreaks**: Role-playing scenarios, hypotheticals
   - "You are an actor playing a villain who..."
   - "In a fictional story, describe how to..."

2. **Prompt injections**: Embedded instructions override safety
   - User input contains hidden instructions
   - "Ignore previous instructions and..."

3. **Adversarial suffixes**: Appended strings that break safety
   - Optimized sequences that maximize harmful output probability
   - Example: [GCG attacks](https://arxiv.org/abs/2307.15043)

4. **Encoding tricks**: Obfuscation bypasses detection
   - ROT13, base64, leetspeak
   - "H0w t0 m4k3..."

5. **Indirect elicitation**: Multi-step reasoning circumvents filters
   - "First explain component A, then component B, now combine..."

**Observation**: Standard RLHF teaches surface-level safety (refuse direct harmful requests) but not robust safety (resist sophisticated attacks).

## Core challenge: Safety without brittleness

**Competing goals**:
- **Safety**: Never produce harmful outputs
- **Capability**: Remain helpful for legitimate use cases
- **Robustness**: Resist adversarial elicitation attempts

**Failure modes**:
- Too strict: False positives (refuse benign requests)
- Too loose: False negatives (allow harmful requests)
- Brittle: Easily bypassed with clever prompts

**Goal**: Find safety strategy that is both effective and robust

## Proposed approaches

### Approach 1: Adversarial training for safety

Train on adversarially generated prompts during RLHF:

**Method**:
1. **Standard RLHF**: Train on normal prompts with safety reward
2. **Adversarial generation**: Generate jailbreak attempts
   - Use red team humans to craft attacks
   - Use automated attack methods (GCG, prompt injection generators)
   - Use LLMs prompted to generate jailbreaks

3. **Adversarial safety training**: Include adversarial prompts in RLHF
   - High penalty for failing on adversarial prompts
   - Policy learns to resist sophisticated attacks

4. **Iterative hardening**: As policy improves, generate stronger attacks
   - Curriculum of increasing adversarial difficulty

**Benefits**:
- Direct training on attack distribution
- Policy learns robust safety, not just surface refusal
- Adapts to new attack strategies

**Challenges**:
- Generating diverse, realistic attacks (coverage problem)
- Balancing adversarial and normal training (don't break capabilities)
- Arms race between attack generation and defense

### Approach 2: Certified robust safety via constraints

Use formal methods to enforce safety constraints:

**Method**:
1. Define safety specification: Properties outputs must satisfy
   - Example: "Never generate code for malware"
   - Formalized as constraints on output space

2. During training, enforce constraints:
   - Constrained RL: Optimize utility subject to safety constraints
   - Reject outputs violating constraints (hard constraint)
   - Or penalize violations (soft constraint)

3. Verification: Prove model satisfies constraints under perturbations
   - Inspired by certified adversarial robustness in vision

**Benefits**:
- Formal guarantees (if achievable)
- Clear safety specification
- Systematic approach

**Challenges**:
- Defining complete safety specifications (hard for open-ended generation)
- Verification is computationally expensive
- May be overly conservative (false positives)

### Approach 3: Intent classification and filtering

Separate understanding user intent from generating response:

**Architecture**:
```
User prompt → Intent classifier → {
  if malicious_intent: Refuse (robust refusal system)
  if benign_intent: Generate response
}
```

**Intent classifier training**:
- Train on diverse malicious prompts (including adversarial examples)
- Robustness training for classifier itself
- High recall (catch attacks) with acceptable precision (minimize false positives)

**Benefits**:
- Modular defense (separate classifier from generator)
- Can update classifier independently
- Specialized model for safety detection

**Challenges**:
- Intent can be ambiguous (legitimate use vs. attack)
- Classifier itself can be attacked
- Latency (additional model call)

### Approach 4: Robust reward modeling

Train reward models that penalize adversarial behaviors:

**Method**:
1. Standard RM: Scores responses for quality
2. Adversarial RM: Specifically detects jailbreak attempts and harmful outputs
3. Combined reward: R_total = R_quality - β·R_adversarial

**Adversarial RM training**:
- Train on dataset of (prompt, response) pairs
- Label adversarial/harmful examples
- RM learns to detect subtle safety violations

**Benefits**:
- Reward model provides robust safety signal for RL
- Can incorporate adversarial detection into optimization
- Flexible (update RM as new attacks emerge)

### Approach 5: Meta-learning for robustness

Train policy to be robust to prompt perturbations:

**Method**:
1. Meta-training objective: Policy should give consistent (safe) outputs under prompt variations
2. For each prompt:
   - Generate perturbations (paraphrases, encodings, etc.)
   - Policy should refuse consistently across all perturbations
3. Regularization: Encourage output consistency under semantically-similar prompt variations

**Benefits**:
- General robustness to distribution shift
- Not specific to known attacks (hopefully generalizes to unknown attacks)
- Principled approach (consistency regularization)

### Approach 6: Multi-layer defense

Combine multiple defense mechanisms:

**Layers**:
1. **Input filtering**: Detect and reject obvious attacks
2. **Robust policy**: Generate safe outputs even if input filter fails
3. **Output filtering**: Catch any harmful outputs before returning to user
4. **Monitoring**: Log and flag suspicious interactions

**Benefits**:
- Defense in depth (no single point of failure)
- Each layer catches different attack types
- Graceful degradation

## Red teaming and evaluation

**Critical**: Evaluation must use realistic attacks

### Red team methodology

**Human red teaming**:
- Hire security researchers and creative writers
- Task: Try to elicit harmful outputs
- Incentivize: Bounties for successful attacks
- Iterative: As model improves, red team adapts

**Automated red teaming**:
- Use LLMs to generate adversarial prompts
- Techniques:
  - Prompted generation: "Generate a jailbreak for..."
  - Optimization: GCG, AutoPrompt, gradient-based attacks
  - Mutation: Modify known attacks to evade detection

**Attack diversity**:
- Ensure coverage across attack types (jailbreaks, injections, encodings, etc.)
- Measure: Attack success rate across diverse strategies

### Evaluation metrics

**Metric 1: Attack success rate (ASR)**
- Fraction of attacks that elicit harmful outputs
- Lower is better

**Metric 2: Benign request false positive rate**
- Fraction of legitimate requests incorrectly refused
- Lower is better (maintain capability)

**Metric 3: Robustness-capability Pareto curve**
- Plot: Safety (1 - ASR) vs. Capability (accuracy on benign tasks)
- Find optimal balance

**Metric 4: Adaptive attack resistance**
- Attacker knows defense mechanism
- Can they craft attacks that still succeed?
- Measures robustness to strongest adversary

## Experimental framework

### Experiment 1: Adversarial training effectiveness

**Setup**:
1. Baseline: Standard RLHF (no adversarial training)
2. Treatment: RLHF + adversarial training
3. Red team both models

**Measure**:
- ASR on held-out attacks
- Capability on benign tasks
- Cost: Additional training time

### Experiment 2: Attack type coverage

**Question**: Which attack types are hardest to defend against?

**Method**:
- Test model against different attack categories
- Measure ASR per category
- Identify weaknesses

### Experiment 3: Transfer to unknown attacks

**Question**: Does training on known attacks improve robustness to novel attacks?

**Method**:
- Train on attack types {A, B, C}
- Test on held-out attack type D
- Measure: Transfer of robustness

### Experiment 4: Defense comparison

**Compare**:
- Adversarial training
- Intent classification
- Robust reward modeling
- Multi-layer defense
- Combinations

**Measure**: Robustness-capability trade-off for each

## Key research questions

1. **Can we achieve robust safety through training?**
   - Or is vulnerability fundamental to LLM architecture?

2. **What's the robustness-capability trade-off?**
   - How much capability must we sacrifice for robustness?
   - Is there a Pareto improvement over current methods?

3. **Do defenses generalize to unknown attacks?**
   - Or do we need continuous adversarial training as attacks evolve?

4. **What attack types are most challenging?**
   - Prioritize defense research

5. **Can formal verification provide guarantees?**
   - Or is empirical red teaming the best we can do?

## Practical deployment

**Staged deployment**:
1. Develop robust policy in controlled setting
2. Red team extensively before deployment
3. Monitor deployed model for attack attempts
4. Continuously update based on observed attacks
5. Maintain feedback loop: Red team → Training → Deployment → Monitoring

**Defense updates**:
- Efficient fine-tuning on new attack types (LoRA, etc.)
- Don't require full retraining when new jailbreak discovered

## Extensions

**Interpretable robustness**:
- Explain why model resists specific attacks
- Help developers understand defense mechanisms

**User-aware robustness**:
- Different robustness levels for different users/use cases
- High-security applications: Maximum robustness
- Creative applications: Lower restrictions

**Collaborative red teaming**:
- Community-driven attack discovery
- Bug bounties for jailbreaks
- Crowdsourced defense improvement

**Robustness certification**:
- Third-party auditing of model robustness
- Standardized test suites for adversarial resistance

**Cross-model robustness transfer**:
- Defenses that transfer across model families
- General principles of robust safety

This research is critical for deploying safe AI systems: without robustness to adversarial prompts, safety training is superficial and easily bypassed, undermining trust and creating risks in deployment.
