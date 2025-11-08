# Multi-agent debate for self-improvement

Recent work like [Self-Refine](https://arxiv.org/abs/2303.17651) and [Constitutional AI](https://arxiv.org/abs/2212.08073) shows that language models can critique and improve their own outputs. However, most approaches use a single model for both generation and critique, which may lead to blindspots where the model cannot identify its own systematic errors.

Multi-agent debate, as explored in [Du et al. (2023)](https://arxiv.org/abs/2305.14325), shows that having multiple agents discuss a problem can improve reasoning. This project explores using multi-agent debate as a training signal for self-improvement.

The proposed approach:
1. Given a prompt, generate N diverse responses from different model variants (e.g., different temperatures, different prompts, or different checkpoints)
2. Have these agents engage in structured debate, where each agent critiques others' responses and defends its own
3. After several rounds of debate, either (a) use majority voting or (b) have a judge select the best response
4. Use this as training data for RL or supervised fine-tuning

Key research questions:
- Does debate lead to better training signals than single-model self-critique?
- How many debate rounds are optimal before diminishing returns?
- Should the debaters be diverse (different models/prompts) or homogeneous?
- Can the policy learn to internalize the debate process over time?

This could be tested on domains like math problem solving, code generation, or creative writing, where there are multiple valid approaches and the debate process can reveal trade-offs.

