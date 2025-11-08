# Meta-learning for rapid preference adaptation

Current RLHF pipelines train a single reward model on aggregated human preferences and use it to train a policy that serves all users. However, individual users have diverse preferences—some prefer concise responses, others detailed explanations; some value creativity, others accuracy. Training separate models for each user is impractical, but serving everyone with the same model ignores valuable personalization opportunities.

Meta-learning, or "learning to learn," has shown success in few-shot adaptation across tasks ([MAML](https://arxiv.org/abs/1703.03400), [Prototypical Networks](https://arxiv.org/abs/1703.05175)). This project explores using meta-learning to train reward models that can quickly adapt to individual user preferences with minimal feedback.

## Motivation

Standard RLHF workflow:
1. Collect thousands of preference labels from many humans
2. Train one reward model on aggregated data
3. Train one policy for all users

Desired personalized workflow:
1. Train a meta-reward model on diverse preference data
2. For each new user, collect 5-20 preference labels
3. Fine-tune reward model to that user in seconds
4. Either fine-tune policy or use reward-guided decoding for personalization

The challenge: how do we train a reward model that can adapt to new preference patterns with very few examples?

## Proposed approach

**Meta-training phase:**

Use Model-Agnostic Meta-Learning (MAML) or similar meta-learning algorithms on preference data:

1. Partition preference data into "user groups" (can be synthetic or based on clustering of annotator preferences)
2. For each meta-training episode:
   - Sample a user group
   - Split their data into support set (few examples) and query set (evaluation)
   - Fine-tune reward model on support set
   - Evaluate on query set
   - Compute meta-gradient through the adaptation process
3. Update meta-parameters to optimize for quick adaptation

**Adaptation phase:**

For a new user:
1. Collect 5-20 pairwise preferences
2. Fine-tune the meta-trained reward model on these preferences (few gradient steps)
3. Use personalized reward model for inference-time guidance or policy fine-tuning

## Alternative: Contextual reward models

Instead of fine-tuning, train the reward model to be contextual:
- Input: (prompt, response, user embedding)
- Output: reward score
- The user embedding is learned from their few-shot preferences
- Use meta-learning to learn good user embeddings quickly

This allows instant personalization without fine-tuning.

## Implementation strategy

Using the [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook):

1. Simulate diverse user preferences by:
   - Using different human annotators as separate "users"
   - Creating synthetic preference variations (e.g., "prefers short responses," "prefers detailed reasoning")
   - Clustering existing annotators by preference patterns

2. Implement MAML or Reptile meta-learning on reward model training
3. Evaluate adaptation performance: how few examples needed to match fully-trained personalized reward model?

## Key research questions

- How many few-shot examples are needed for meaningful personalization?
- Does meta-learning outperform simple fine-tuning from a general reward model?
- Can we identify preference dimensions that are easy vs. hard to adapt?
- Does personalization improve user satisfaction in practice, or do users have inconsistent preferences?
- How do we handle distribution shift when user preferences evolve over time?

## Evaluation

- **Quantitative**: Train personalized reward models for held-out annotators using various amounts of their data (1, 5, 10, 20 examples). Measure accuracy on their remaining preferences.
- **Qualitative**: Human evaluation studies where users provide a few preferences and rate personalized vs. generic model outputs.
- **Efficiency**: Compare meta-learning vs. fine-tuning from scratch in terms of adaptation speed and data efficiency.

## Connection to broader goals

Personalization is crucial for deploying assistant models that serve diverse user populations. This approach could enable:
- Individual users customizing their AI assistant with minimal effort
- Enterprise deployments with company-specific preference tuning
- Culturally-adapted models that respect different norms and values
- A/B testing new preference patterns without full retraining

If successful, this could make RLHF more flexible and user-centric while maintaining efficiency.
