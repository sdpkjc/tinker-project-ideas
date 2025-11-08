# Cross-lingual preference transfer for multilingual RLHF

RLHF requires large amounts of preference data, which is expensive to collect even in high-resource languages like English. For the hundreds of other languages, collecting sufficient preference data is often impractical or impossible. This creates a gap: English models benefit from extensive RLHF while other language models lag behind.

This project explores cross-lingual preference transfer: using preference data from high-resource languages (primarily English) to improve models in other languages, reducing the need for language-specific preference collection.

## Problem: Preference data scarcity for non-English languages

**Current situation**:
- English RLHF: 100K+ preference labels available
- Spanish, French, German: ~1K-10K labels
- Long-tail languages: <100 labels or none

**Consequences**:
- English models are highly capable and aligned
- Non-English models lack post-training benefits
- Global inequality in AI capabilities
- Most users don't speak English

**Question**: Can we transfer preference knowledge from English to other languages?

## Core hypothesis

**Key assumption**: Quality preferences are partially language-universal

**Universal aspects**:
- Factual correctness
- Logical coherence
- Helpfulness for user's goal
- Safety (no harmful content)
- Structure and formatting

**Language-specific aspects**:
- Idiom and cultural appropriateness
- Formality and register conventions
- Language-specific style preferences

**Hypothesis**: If most preferences are universal, we can transfer from English and fine-tune on small language-specific data.

## Proposed approaches

### Approach 1: Translate-and-train

**Method**:
1. Take English preference dataset
2. Translate prompts and responses to target language
3. Assume preferences transfer (if A > B in English, then A_translated > B_translated)
4. Train reward model on translated preferences
5. Use for RLHF in target language

**Benefits**:
- Simple and straightforward
- Leverages existing English data
- Scalable to many languages

**Challenges**:
- Translation quality (errors propagate)
- Cultural misalignment (English preferences may not apply)
- Translation artifacts (unnatural phrasing)

**Mitigation**:
- Use high-quality translation models
- Mix with small amount of native language data
- Filter poor translations

### Approach 2: Multilingual reward model

**Method**:
1. Train single reward model on multilingual data
2. Use multilingual base model (mBERT, XLM-R, mT5)
3. Mix English preferences (abundant) with other languages (scarce)
4. Model learns shared preference concepts across languages
5. Transfer from high-resource to low-resource

**Benefits**:
- Direct cross-lingual transfer through shared representations
- Single model for all languages (simpler deployment)
- Leverages multilingual model's cross-lingual alignment

**Training strategy**:
- 90% English preferences, 10% mixed other languages
- Or curriculum: Start English-only, gradually add other languages
- Or multi-task: Language-specific heads + shared trunk

### Approach 3: Zero-shot cross-lingual transfer

**Method**:
1. Train English reward model on English data
2. Apply directly to non-English prompts/responses
3. Relies on multilingual base model's cross-lingual capabilities
4. No additional training on target language

**Test**: How well does English RM transfer zero-shot?

**Expected**: Partial transfer (universal preferences work, language-specific fail)

**Use case**: Quick prototyping before collecting target language data

### Approach 4: Few-shot adaptation

**Method**:
1. Start with English RM or multilingual RM
2. Collect small amount of target language preferences (50-500 examples)
3. Fine-tune RM on target language data
4. Adaptation focuses on language-specific aspects

**Benefits**:
- Best of both worlds: Leverage English data + adapt to target language
- Efficient use of limited target language annotations
- Addresses cultural and linguistic differences

**Similar to**: Few-shot domain adaptation (applied to language instead of domain)

### Approach 5: Preference alignment through parallel data

**Method**:
Use parallel prompts/responses (same meaning, different languages):

1. Create parallel dataset:
   - Prompt in English + translation to target language
   - Response in English + translation to target language
   - Preference label (should be same across languages)

2. Train RM with contrastive loss:
   - Align embeddings for parallel prompts/responses
   - Encourage consistent preferences across languages

**Benefits**:
- Explicitly models cross-lingual alignment
- Grounded in parallel examples
- Learns what aspects transfer vs. what's language-specific

## Evaluation methodology

### Metric 1: Cross-lingual transfer effectiveness

**Setup**:
1. Train English RM on English preferences
2. Apply to target language using each approach
3. Collect ground-truth preferences in target language for evaluation
4. Measure: RM accuracy on target language test set

**Baselines**:
- Zero-shot transfer (no target language data)
- Train from scratch on target language only (upper bound with infinite data)

### Metric 2: Data efficiency

**Question**: How much target language data is needed?

**Method**:
- Start with English RM
- Vary amount of target language data for adaptation: 10, 50, 100, 500, 1000
- Plot: Accuracy vs. target language data size

**Compare**: Transfer + adaptation vs. training from scratch

### Metric 3: Policy quality after RLHF

**Critical test**: Does transferred RM produce good policies?

**Method**:
1. Use transferred/adapted RM for RLHF in target language
2. Evaluate policy with human evaluators who speak target language
3. Compare to: Policy trained with native RM (if available)

### Metric 4: Cultural appropriateness

**Question**: Do transferred preferences respect target language culture?

**Method**:
- Identify culture-specific scenarios (greetings, humor, taboos, formality)
- Test RM on these scenarios
- Human evaluation by native speakers: Does model respect cultural norms?

**Expected**: Universal preferences (correctness) transfer well, cultural preferences need target language data

### Metric 5: Translation quality impact

**Question**: How much does translation quality affect transfer?

**Method**:
- Use high-quality vs. low-quality translations
- Measure RM accuracy with each
- Identify: What translation errors most hurt transfer?

## Experimental framework

### Experiment 1: Language similarity effects

**Question**: Does transfer work better for similar languages?

**Test**:
- English → {Spanish, French, German} (related languages)
- English → {Chinese, Arabic, Japanese} (distant languages)
- Measure transfer effectiveness for each

**Hypothesis**: Related languages transfer better

### Experiment 2: Preference universality

**Question**: Which preferences transfer well vs. poorly?

**Method**:
1. Categorize preferences:
   - Factual correctness
   - Logical reasoning
   - Helpfulness
   - Safety
   - Style and tone
   - Cultural appropriateness

2. Measure transfer success for each category

**Expected**: Factual/logical transfer well, cultural transfers poorly

### Experiment 3: Multilingual vs. translation

**Compare**:
- Translate English data to target language, train mono-lingual RM
- Train multilingual RM on mixed languages

**Question**: Which approach better?

### Experiment 4: Low-resource language transfer

**Setup**:
- Select truly low-resource language (e.g., Swahili, Bengali)
- Collect small gold preference dataset (500 examples)
- Test all transfer approaches

**Question**: Can we enable RLHF for low-resource languages?

## Key research questions

1. **Do preferences transfer across languages?**
   - Which aspects transfer? Which don't?
   - How universal vs. language-specific are quality judgments?

2. **What's the best transfer method?**
   - Translation, multilingual model, few-shot adaptation, or combination?

3. **How much target language data is needed?**
   - Can we achieve 80% of native RM quality with 10% of data?

4. **Does transferred RM produce good policies?**
   - Not just RM accuracy, but RLHF outcome?

5. **Can we enable RLHF for low-resource languages?**
   - Make RLHF accessible globally?

## Practical considerations

**Infrastructure**:
- High-quality translation system
- Multilingual base models
- Cross-lingual evaluation benchmarks

**Data collection strategy**:
- Prioritize collecting data in languages with no existing preferences
- Use transfer to bootstrap, then refine with target language data

**Deployment**:
- Single multilingual RM vs. language-specific RMs?
- Trade-offs in simplicity vs. specialization

## Extensions

**Cross-cultural preference modeling**:
- Explicitly model cultural differences in preferences
- Allow different preference weights for different cultures

**Active learning for cross-lingual transfer**:
- Identify which target language examples most valuable to collect
- Focus on cases where transfer fails

**Continual cross-lingual learning**:
- As we collect more data in various languages, improve transfer
- Meta-learn what transfers well vs. poorly

**Cross-lingual preference synthesis**:
- Use LLMs to generate synthetic preferences in target language
- Bootstrap from English, adapt to target language

**Multi-source transfer**:
- Transfer from multiple source languages, not just English
- Ensemble of RMs from different languages

**Language-agnostic preference learning**:
- Learn preference concepts independent of language
- Apply to any language without language-specific training

This research could democratize RLHF, making advanced post-training accessible for the billions of people who don't speak English, reducing global AI inequality and improving models for diverse linguistic and cultural communities.
