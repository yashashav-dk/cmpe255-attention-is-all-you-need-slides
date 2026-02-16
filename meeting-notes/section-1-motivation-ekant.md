# Section 1: Motivation & Big Picture

**Presenter:** Ekant Kapgate
**Slides:** 1-7 (Title, Section Divider, The Sequence Problem, Problems with RNNs, The Key Insight, Transformer Overview, Roadmap)

---

## Slide 1 — Title Slide

- Introduce the paper: "Attention Is All You Need" by Vaswani et al., NeurIPS 2017
- Mention all team members
- Course: CMPE 257 — Machine Learning, Prof. Gautam Krishna

## Slide 2 — Section Divider: Motivation & Big Picture

- Transition line: "Why do we need the Transformer?"
- Set the stage — before 2017, sequence tasks were dominated by RNNs

## Slide 3 — The Sequence Problem

**Key points to cover:**
- Before 2017, sequence tasks (translation, summarization, etc.) relied on Recurrent Neural Networks (RNNs)
- Walk through the RNN diagram: tokens are processed one at a time (t=1, t=2, ... t=N)
- Each step passes a hidden state to the next — this is the "recurrence"
- **Key bottleneck:** Each step must wait for the previous hidden state — inherently sequential, no parallelism
- This means you can't take advantage of modern GPU hardware which excels at parallel computation

**Talking point:** "Think of it like a factory assembly line where each worker must wait for the previous one to finish — even if the tasks could be done simultaneously."

## Slide 4 — Problems with RNNs

**Three core problems:**
1. **Slow** — Sequential processing means you can't use GPU parallelism. Training time scales linearly with sequence length.
2. **Vanishing/exploding gradients** — Gradients flow through every time step. For long sequences, they either shrink to near-zero (vanishing) or blow up (exploding). This makes training unstable.
3. **Long-range dependencies** — Information from early tokens must survive through every hidden state. RNNs struggle to remember context from far back in the sequence.

**Important note:** LSTMs and GRUs improved problems (2) and (3) with gating mechanisms, but they did NOT solve problem (1) — they're still fundamentally sequential.

## Slide 5 — The Key Insight

**The core question the paper asks:**
> "What if every token could attend to every other token — in parallel — without recurrence?"

**Comparison to drive home:**
- **RNN:** Token 1 -> Token 2 -> ... -> Token N (path length O(N)) — information must travel through every step
- **Transformer:** Every token <-> Every token (path length O(1)) — direct connections between all tokens

**Key insight:** Attention replaces recurrence entirely. All tokens are processed at once, enabling massive GPU parallelization.

**Talking point:** "Instead of passing information down a chain, every word can directly talk to every other word — simultaneously."

## Slide 6 — Transformer Overview

**Walk through the high-level architecture diagram:**

**Encoder (left, x6 layers):**
- Input Embedding + Positional Encoding
- Multi-Head Self-Attention
- Add & Norm (residual connection + layer normalization)
- Feed-Forward Network
- Add & Norm
- Produces Encoder Output

**Decoder (right, x6 layers):**
- Output Embedding + Positional Encoding
- Masked Multi-Head Attention (can only see previous tokens)
- Add & Norm
- Cross-Attention (Q from decoder, K/V from encoder) — this is the bridge
- FFN + Add & Norm
- Linear -> Softmax (produces output probabilities)

**The dashed K,V arrow:** Shows how encoder output feeds into the decoder via cross-attention.

**Talking point:** "The encoder reads the entire input at once. The decoder generates output one token at a time, but it can look back at the full encoder output through cross-attention."

## Slide 7 — Roadmap

**Preview the presentation structure and hand-off points:**
1. **Motivation** (Ekant) — Why Transformers? High-level view *(just covered)*
2. **Encoder** (Saransh) — Embeddings, positional encoding, self-attention
3. **Decoder** (Vineet) — Masked attention, training vs inference
4. **Impact** (Yashashav) — BERT, GPT, and beyond

**Transition:** "Now that we understand WHY we need the Transformer, Saransh will walk us through HOW the encoder works."

---

## Tips for Delivery

- Start with a relatable example: "When you type a sentence into Google Translate, how does the model understand the whole sentence?"
- Use the RNN diagram to build intuition — trace the sequential flow
- Make the RNN vs Transformer comparison vivid — O(N) vs O(1) path length
- End with a clear hand-off to Saransh for the Encoder section
- Estimated time: ~5-6 minutes
