# Section 3: Decoder & Training/Inference

**Presenter:** Vineet Malewar
**Slides:** 17-25 (Section Divider, Decoder Overview, Masked Attention, Cross-Attention, Linear + Softmax, Training: Teacher Forcing, Autoregressive Inference, Inference Strategies, The Full Picture)

---

## Slide 17 — Section Divider: Decoder & Training

- Transition line: "How the model generates output"
- The decoder takes the encoder's representations and produces the target sequence

## Slide 18 — Decoder Overview

**Key difference from encoder:** The decoder has **three** sub-layers per block (vs. two in encoder):

**The flow:** Target -> Masked Self-Attn -> Cross-Attn -> FFN -> Add&Norm

**The three sub-layers explained:**

| Sub-layer | Purpose | Q, K, V |
|-----------|---------|---------|
| **Masked Self-Attn** | Attend to previous output tokens only | All from decoder |
| **Cross-Attention** | Attend to encoder output | Q=decoder, KV=encoder |
| **Feed-Forward** | Non-linear transformation | — |

**Talking point:** "The encoder reads everything at once, but the decoder must be careful — it can only look at what it has generated so far, not future tokens."

## Slide 19 — Masked Attention

**Why masking is necessary:**
- The decoder must be **causal** — position i can only see positions < i
- Without masking, the model could "cheat" by looking at future tokens during training

**Walk through the two matrices:**

**Before masking (left):**
- Full attention matrix — every token sees every other token
- YOUR sees CAT and IS (which it shouldn't during generation)

**After masking (right):**
- Upper triangle set to 0
- YOUR: only attends to itself (1.00)
- CAT: attends to YOUR (0.47) and itself (0.53)
- IS: attends to all previous tokens (0.31, 0.34, 0.35)

**How it works technically:** Future positions are set to **-infinity** before softmax -> they become **0** after softmax. No information leaks from future tokens.

**Talking point:** "Imagine reading a book where you can only see the pages you've already read — that's what masking enforces."

## Slide 20 — Cross-Attention

**The bridge between encoder and decoder — the most important mechanism for translation.**

**Walk through the diagram:**
- **Encoder Output** (seq, 512) provides **K** (Keys) and **V** (Values)
- **Decoder State** (seq, 512) provides **Q** (Queries)
- Cross-Attention computes: softmax(QK^T / sqrt(d_k)) * V
- Output goes to the next layer

**Intuition:**
- **Q** from decoder = "What am I looking for in the source?"
- **K** from encoder = "What does each source word contain?"
- **V** from encoder = "What information should I extract?"

**Key insight:** Each output token attends to the relevant parts of the input. For example, when generating "molto" (Italian for "much"), the decoder attends strongly to "much" in the encoder output.

## Slide 21 — Linear Layer & Softmax

**The final transformation that produces actual word predictions:**

**The flow:** Decoder Out (seq, 512) -> Linear (512 -> vocab) -> Softmax (vocab_size)

**Step by step:**
1. **Linear layer** projects d_model (512) to vocabulary size (e.g., 37K tokens) — produces raw scores (logits)
2. **Softmax** converts logits to probabilities that sum to 1
3. **During training:** Compare predicted distribution with ground truth via **cross-entropy loss**
4. **During inference:** Select the highest-probability token (or use beam search)

**Talking point:** "The linear layer is essentially asking 'which of the 37,000 words in our vocabulary is most likely next?'"

## Slide 22 — Training: Teacher Forcing

**Example:** "I love you very much" -> "Ti amo molto"

**Walk through the diagram:**
- **Encoder** processes the source sentence ("I love you very much") — produces K, V
- **Decoder** receives the shifted target ("<SOS> Ti amo molto") — the target shifted right by one position
- **Linear + Softmax** produces predictions: "Ti amo molto <EOS>"
- **Cross-Entropy Loss** compares predictions vs. ground truth "Ti amo molto <EOS>"

**Critical insight — "All in ONE time step!":**
- During training, the decoder sees the **entire target** (shifted right) at once
- This is possible because of masking — each position can only attend to previous positions
- This is called **teacher forcing** — we feed the correct answers as input rather than the model's own predictions
- Massive speedup over autoregressive training

**Talking point:** "During training, we give the decoder the 'answer key' as input — but masking ensures it can't peek ahead. This lets us train on all positions in parallel."

## Slide 23 — Autoregressive Inference

**Unlike training, inference generates tokens one at a time:**

| Step | Decoder Input | Output Token |
|------|--------------|-------------|
| 1 | \<SOS\> | Ti |
| 2 | \<SOS\> Ti | amo |
| 3 | \<SOS\> Ti amo | molto |
| 4 | \<SOS\> Ti amo molto | \<EOS\> |

**Key insight:** The **encoder runs once** — its output is reused at every step. Only the decoder runs repeatedly, appending each new token to its input.

**Talking point:** "This is why inference is slower than training — we can't parallelize the generation. Each word depends on all previous words."

## Slide 24 — Inference Strategies

**Two approaches to selecting tokens:**

**Greedy Decoding (left):**
- Always pick the top-1 (highest probability) token at each step
- Fast but can miss better overall sequences
- Gets stuck in local optima
- Example: step 1: argmax -> "Ti", step 2: argmax -> "amo", step 3: argmax -> "molto"

**Beam Search with B=3 (right):**
- Keep top B candidates at each step
- Explore multiple paths in parallel
- Pick the best complete sequence at the end
- Example: step 1: ["Ti", "Il", "Io"], step 2: ["Ti amo", "Ti vog.."], step 3: pick best

**Talking point:** "Greedy is like always taking the nearest exit on the highway — you might miss a faster route. Beam search explores multiple routes simultaneously."

## Slide 25 — The Full Picture

**Complete encoder-decoder architecture in one diagram:**

**Left column (Encoder):**
- Source: "I love you" feeds into Input Embedding + PE
- Self-Attention -> Add & Norm -> Feed-Forward -> Add & Norm
- x6 layers -> Encoder Output

**Right column (Decoder):**
- Target: "<SOS> Ti amo" feeds into Output Embedding + PE
- Masked Self-Attn -> Add & Norm -> Cross-Attention -> Add&Norm + FFN
- x6 layers -> Linear + Softmax -> "Ti amo molto <EOS>"

**The dashed K,V arrow:** Encoder output feeds Keys and Values into the decoder's Cross-Attention layer.

**Transition:** "That's the complete Transformer architecture. Now Yashashav will show us why this paper changed everything — from NLP to vision, biology, and beyond."

---

## Tips for Delivery

- This section covers the most diverse set of concepts — masked attention, cross-attention, training, inference
- The training vs. inference distinction is crucial — emphasize that training is parallel (teacher forcing) but inference is sequential (autoregressive)
- For cross-attention, use the Q/K/V analogy: "The decoder asks questions (Q), and the encoder provides the catalog (K) and the answers (V)"
- The Full Picture slide is a great recap moment — briefly trace the full flow from input to output
- Estimated time: ~8-10 minutes
