# Section 2: The Encoder

**Presenter:** Saransh Soni
**Slides:** 8-16 (Section Divider, Input Embeddings, Positional Encoding, Why Sine & Cosine, Self-Attention Intuition, Self-Attention Steps, Multi-Head Attention, Layer Norm & FFN, Encoder Stack)

---

## Slide 8 — Section Divider: The Encoder

- Transition line: "From raw tokens to contextualized representations"
- The encoder's job: take a sequence of words and produce rich, context-aware vector representations

## Slide 9 — Input Embeddings

**Key points:**
- Raw text must become numbers the model can work with
- Pipeline: Tokens -> Vocab IDs -> Embedding vectors (512 dimensions)
- Example on slide:
  - YOUR -> ID 105 -> [952, 5450, 1853, ...] (512d vector)
  - CAT -> ID 6587 -> [621, 1304, 0.6, ...] (512d vector)
  - IS -> ID 5475 -> [776, 5567, 58.9, ...] (512d vector)
- Each token maps to a learned vector of size d_model = 512
- These embeddings capture **meaning** but NOT position — the model doesn't yet know word order

**Talking point:** "Similar words end up with similar embedding vectors — 'cat' and 'dog' would be close in this 512-dimensional space."

## Slide 10 — Positional Encoding

**Key points:**
- Since there's no recurrence, the model has no built-in sense of word order
- Solution: **inject position info** explicitly by adding a positional encoding vector

**The formula:**
- PE(pos, 2i) = sin(pos / 10000^(2i/d))
- PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

**The flow:** Embedding + Positional Encoding = Encoder Input

**Three key properties:**
1. Computed **once**, reused for all sentences (not learned, deterministic)
2. Sine/cosine at different frequencies -> unique pattern per position
3. Model can learn **relative positions**: PE(pos+k) is a linear function of PE(pos)

**Talking point:** "Think of it like giving each word a unique 'address' in the sentence, so the model knows that 'cat' is the second word, not the fifth."

## Slide 11 — Why Sine & Cosine?

**Four reasons for choosing sinusoidal functions:**
1. **Continuous** — smooth functions the model can learn from (vs. discrete one-hot)
2. **Bounded** — always between -1 and 1, which keeps training stable
3. **Unique per position** — different frequencies across dimensions create a unique fingerprint
4. **Relative positions** — PE(pos+k) is a linear transform of PE(pos), so the model can learn relative distances

**The frequency diagram (right side):**
- High frequency (i=0): oscillates rapidly — captures fine-grained position
- Low frequency (i=255): changes slowly — captures coarse position
- "Like a binary clock" — different dimensions tick at different rates

## Slide 12 — Self-Attention: Intuition

**Goal:** Let each word "look at" every other word to capture relationships.

**The core formula:**
> Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

- In **self-attention**: Q = K = V = the input sequence itself
- Each word produces a Query ("what am I looking for?"), Key ("what do I contain?"), and Value ("what info do I provide?")

**The attention matrix on the slide:**
- 6x6 matrix for "YOUR CAT IS A LOVELY CAT"
- Each row sums to 1 (softmax normalization)
- Diagonal tends highest — each word attends most to itself
- But off-diagonal values show cross-word relationships

**Talking point:** "When processing the word 'lovely', the model learns to pay extra attention to 'cat' — because 'lovely' is modifying 'cat'."

## Slide 13 — Self-Attention Steps

**Walk through the 4 steps:**

**Setup:** Q (seq, d_k) x K^T (d_k, seq) = Scores (seq, seq)

1. **Compute scores:** QK^T — dot product measures similarity between every pair of tokens
2. **Scale:** Divide by sqrt(d_k) — prevents softmax from getting too peaked (saturated) when d_k is large
3. **Softmax:** Normalize each row to get attention weights (sum to 1)
4. **Weighted sum:** Multiply by V — each output is a weighted mix of all value vectors

**Key insight:** Each output captures the token's meaning PLUS its interaction with every other token.

**Talking point:** "The scaling by sqrt(d_k) is a small but critical detail — without it, the dot products become too large and softmax outputs near-zero gradients."

## Slide 14 — Multi-Head Attention

**The formula:**
> MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O

Where each head_i = Attention(Q * W_i^Q, K * W_i^K, V * W_i^V)

**Why multiple heads?**
- Each head learns **different relationships** — one might capture syntax, another semantics, another coreference
- Like multiple "perspectives" on the same data

**Dimensions:**
| Parameter | Value |
|-----------|-------|
| Heads (h) | 8 |
| d_model | 512 |
| d_k = d_v = d_model/h | 64 |

**Key insight:** Split -> Attend in parallel -> Concat -> Project. Same computational cost as single-head with full dimensions.

**Talking point:** "One head might learn 'what adjective describes this noun?', another might learn 'what verb goes with this subject?'"

## Slide 15 — Layer Norm & FFN

**Two components that complete each encoder sub-layer:**

**Layer Normalization (left column):**
- Formula: x_hat = (x - mean) / sqrt(variance + epsilon)
- Normalizes across **features** (not batch) — more stable for sequence data
- Learnable gamma (scale) and beta (shift) parameters
- Stabilizes deep network training

**Feed-Forward Network (right column):**
- FFN(x) = ReLU(xW1 + b1)W2 + b2
- Two linear layers with ReLU activation in between
- Inner dimension: 2048 (4x d_model)
- Applied **per position independently** — same transformation for every token

**The flow diagram:**
Input -> MH-Attn -> Add&Norm -> FFN -> Add&Norm -> Out

**Important:** Residual connections (the "Add" part) help gradient flow through the 6 stacked layers. This pattern repeats N=6 times.

## Slide 16 — The Encoder Stack

**The full picture of the encoder:**
- Input Embeddings + Positional Encoding feed into Encoder Layer 1
- Layer 1 -> Layer 2 -> ... -> Layer 6
- Each layer contains: Multi-Head Attn -> Add & Norm -> Feed-Forward -> Add & Norm
- Final output: Encoder Layer 6 produces "Encoder Output" — a (seq, 512) matrix

**Key insight:** For each token, the output is a 512-dimensional vector encoding meaning, position, AND relationships with all other tokens. This is passed to the decoder.

**Transition:** "Now that we have these rich, contextualized representations from the encoder, Vineet will explain how the decoder uses them to generate output."

---

## Tips for Delivery

- This is the most technically dense section — pace yourself
- Use the "YOUR CAT IS A LOVELY CAT" example consistently across slides 12-13
- On the attention formula, explain Q/K/V with an analogy: "Q is like a search query, K is like a page title, V is like the page content"
- For multi-head attention, the "multiple perspectives" analogy works well
- Don't rush through Layer Norm & FFN — the residual connections are crucial for understanding why deep Transformers train successfully
- Estimated time: ~8-10 minutes
