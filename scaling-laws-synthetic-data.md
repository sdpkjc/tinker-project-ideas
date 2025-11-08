# Scaling laws for synthetic data quality and quantity

Self-generated synthetic data is increasingly important for LLM training, as seen in [Phi-1.5](https://arxiv.org/abs/2309.05463), [Self-Instruct](https://arxiv.org/abs/2212.10560), and various distillation techniques. However, there's limited systematic understanding of how synthetic data quality and quantity trade off.

The key question: given a fixed compute budget for generating synthetic data, should you generate a large amount of lower-quality data or a smaller amount of higher-quality data?

Proposed experimental framework:
1. Fix a task domain (e.g., math problems, coding, instruction following)
2. Create a spectrum of data generation configurations that vary quality vs. quantity:
   - Low quality, high quantity: single-pass generation from a weaker model
   - Medium quality, medium quantity: best-of-N sampling from a medium model
   - High quality, low quantity: iterative refinement with a strong model
3. Control for total generation compute across all configurations
4. Train models on each synthetic dataset and evaluate downstream performance

Extensions:
- Investigate optimal curriculum: does starting with high-quantity low-quality data and transitioning to low-quantity high-quality data work better than either extreme?
- Study how the quality-quantity trade-off changes with model scale
- Explore mixtures: what's the optimal ratio when combining datasets of different quality levels?
- Compare supervised learning vs. RL on synthetic data of varying quality

This would provide practical guidance for practitioners on how to allocate their data generation budget and could reveal fundamental principles about how models learn from imperfect demonstrations.
