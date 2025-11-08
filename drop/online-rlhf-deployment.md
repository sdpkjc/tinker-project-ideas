# Online RLHF: Continual learning from deployment feedback

Standard RLHF is an offline process: collect preferences, train reward model, train policy, then deploy. Once deployed, the policy is static—it doesn't learn from new user interactions. However, deployment generates vast amounts of implicit and explicit feedback that could be used for continual improvement.

This project explores online RLHF: systems that continuously learn from deployment feedback, adapting to user needs and fixing failures in real-time, inspired by contextual bandits ([Agarwal et al., 2014](https://arxiv.org/abs/1402.0555)) and online learning paradigms.

## Motivation

**Current offline RLHF**:
```
Collect preferences → Train RM → Train policy → Deploy (frozen)
```

**Limitations**:
- Can't adapt to new user preferences or domains
- Can't fix newly discovered failure modes
- Expensive retraining cycle to incorporate new data
- Distribution shift between training and deployment

**Desired online RLHF**:
```
Deploy policy → Collect feedback → Update policy → Repeat continuously
```

**Benefits**:
- Adapt to distribution shift in real-time
- Learn from diverse deployment scenarios
- Quickly fix failures without full retraining
- Personalize to user populations over time

## Challenges and risks

**Challenge 1: Feedback quality**
- User feedback is noisy and biased (users complain about errors but rarely praise good outputs)
- Implicit feedback (clicks, dwell time) is ambiguous

**Challenge 2: Distribution shift**
- Updating policy changes the data distribution
- Non-stationarity complicates learning
- Risk of feedback loops and instability

**Challenge 3: Safety**
- Can't deploy untested updates to production
- Need safeguards against adversarial feedback or gaming
- Balance exploration (trying new behaviors) with exploitation (serving users well)

**Challenge 4: Efficiency**
- Can't do full RL training in real-time
- Need lightweight, incremental update methods
- Computational constraints in production

## Proposed approaches

### Approach 1: Reward model continual learning

Update the reward model online, then periodically retrain policy:

**Online RM updates**:
1. Collect user feedback continuously (thumbs up/down, preferences, ratings)
2. Update reward model with incremental learning
   - Use experience replay to prevent catastrophic forgetting
   - Maintain buffer of past preferences
   - Mix new feedback with historical data

3. Periodically retrain policy with updated RM (e.g., daily or weekly)

**Advantages**:
- RM updates are fast and lightweight
- Policy retraining is controllable and testable
- Decouples feedback collection from policy deployment

### Approach 2: Online policy fine-tuning

Directly update the deployed policy using online feedback:

**Method**:
1. Collect user feedback on deployed responses
2. Convert feedback to training signal:
   - Positive feedback → SFT on that response
   - Negative feedback → RL penalty or inverse SFT
   - Preference comparisons → Direct Preference Optimization (DPO)

3. Perform lightweight policy updates (e.g., LoRA fine-tuning)
4. Test updated policy on canary users before full deployment

**Advantages**:
- Direct policy improvement without RM bottleneck
- Can be very responsive (hourly updates possible)
- Simpler pipeline than two-stage RM+policy approach

### Approach 3: Contextual bandit formulation

Treat each deployment interaction as a bandit episode:

**Setup**:
- Context: User prompt + user features
- Actions: Different policy variants or decoding strategies
- Reward: User feedback (explicit or implicit)

**Algorithm**:
- Use contextual bandit algorithms (LinUCB, Thompson Sampling)
- Balance exploration (try different policies) and exploitation (use best policy)
- Learn which policy works best for which contexts

**Advantages**:
- Principled exploration-exploitation trade-off
- Well-studied algorithms with theoretical guarantees
- Natural A/B testing framework

### Approach 4: Mixture of experts with online routing

Maintain multiple policy variants (experts) and learn routing:

**Setup**:
- Train diverse policies (different training data, hyperparameters, or checkpoints)
- Deploy all as mixture of experts
- Learn routing function online: which expert for which prompt?

**Online learning**:
- User feedback trains routing function
- Route similar prompts to best-performing expert
- Periodically retire poorly-performing experts and add new ones

**Advantages**:
- Flexible and modular
- Can add specialized experts for domains discovered during deployment
- Limits risk (single bad expert doesn't break entire system)

## Feedback sources

**Explicit feedback**:
- Binary ratings (thumbs up/down)
- Preference comparisons (regenerate and compare)
- Detailed ratings (1-5 stars on multiple dimensions)
- Text feedback and bug reports

**Implicit feedback**:
- Dwell time (how long user reads response)
- Follow-up behavior (do they accept the suggestion?)
- Task completion (in agentic settings, did they accomplish goal?)
- Session patterns (do they keep using the system?)

**Hybrid feedback**:
- Ask for feedback on subset of interactions
- Infer labels for others using learned correlation
- Active learning: request feedback on high-value examples

## Safety and robustness mechanisms

**Safeguard 1: Staged rollout**
- Test updates on small canary user group
- Monitor metrics (error rates, user satisfaction, safety incidents)
- Gradually expand if metrics are positive
- Rollback if problems detected

**Safeguard 2: Reward model ensemble**
- Maintain multiple reward models (online-updated + stable baseline)
- Only deploy updates if all RMs agree improvement is likely
- Prevents overreaction to noisy feedback

**Safeguard 3: Conservatism constraints**
- Limit KL divergence from stable checkpoint per update
- Prevents drastic policy changes from noisy feedback
- Ensures updates are incremental and reversible

**Safeguard 4: Adversarial robustness**
- Detect and filter adversarial or gaming feedback
- Rate-limit updates from single users
- Monitor for coordinated manipulation attempts

**Safeguard 5: Human oversight**
- Alert human operators to significant policy changes
- Require approval for updates that exceed thresholds
- Manual review of failure cases

## Experimental framework

### Experiment 1: Simulated online RLHF

Before deploying to real users, simulate online learning:

**Setup**:
1. Start with initial policy trained on existing preferences
2. Simulate deployment: Generate responses to test prompts
3. Simulate feedback: Use held-out human preferences or ground truth
4. Update policy using online algorithm
5. Repeat for many iterations

**Measure**:
- Does performance improve over iterations?
- How stable is learning?
- Sample efficiency compared to offline retraining

### Experiment 2: Closed-loop user study

Small-scale user study with real humans:

**Setup**:
1. Deploy policy to small user group (~100 users)
2. Collect feedback over 2-4 weeks
3. Update policy weekly using online RLHF
4. Measure user satisfaction and task success over time

**Compare**:
- Static policy (no updates)
- Online-updated policy
- Periodically-retrained policy (offline)

### Experiment 3: Domain shift adaptation

Test ability to adapt to new domains:

**Setup**:
1. Train policy on domain A (e.g., general Q&A)
2. Deploy to domain B (e.g., coding help)
3. Measure: How quickly does online RLHF adapt to domain B?

**Baseline**: Retraining from scratch on domain B data

## Key research questions

1. **Is online RLHF stable?**
   - Does continual learning converge or diverge?
   - What factors affect stability (update frequency, learning rate, feedback quality)?

2. **How sample-efficient is online learning?**
   - How much deployment feedback is needed to see improvement?
   - Compare to offline RLHF with same amount of data

3. **Can we adapt to distribution shift?**
   - If user prompts change over time, can online RLHF track?
   - How quickly does adaptation occur?

4. **What feedback signals are most valuable?**
   - Explicit vs. implicit feedback
   - Binary ratings vs. detailed preferences
   - Cost-benefit of different feedback collection methods

5. **How do we ensure safety?**
   - What safeguards are necessary and sufficient?
   - Can we detect and prevent failure modes before they impact users?

## Practical deployment considerations

**Infrastructure requirements**:
- Fast inference + lightweight fine-tuning (LoRA, prompt tuning)
- Feedback collection and storage pipeline
- A/B testing and monitoring infrastructure
- Rollback mechanisms

**Operational challenges**:
- Versioning (which policy version did user interact with?)
- Reproducibility (non-deterministic updates complicate debugging)
- Evaluation (how to measure improvement in non-stationary setting?)

**Cost-benefit analysis**:
- Added complexity vs. performance gains
- Infrastructure costs vs. user satisfaction improvements

## Extensions

**Personalized online RLHF**:
- Learn user-specific policies or routing
- Balance personalization with shared learning

**Multi-task online learning**:
- Deploy across multiple tasks simultaneously
- Transfer learning between tasks during online updates

**Federated online RLHF**:
- Learn from multiple deployments without centralizing data
- Privacy-preserving online learning

**Meta-learning for fast adaptation**:
- Pre-train policy to be good at online learning
- Few-shot adaptation to new deployment distributions

This project addresses the gap between static deployed models and the dynamic, ever-changing nature of real-world deployment, potentially enabling continuously-improving AI systems that adapt to users in real-time.
