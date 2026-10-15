# Pre-Generation Circuit Breaker: Stopping LLM Hallucinations via Mid-Layer Linear Probes

> **TL;DR:** Small language models love to guess when a requested piece of information is missing from the prompt. Instead of relying on prompt engineering or running a second "LLM-as-a-Judge" model after the fact, this repo explores a simpler idea: reading the model's mid-layer hidden states at the very end of the prefill stage, right before it generates a single token. Using a lightweight logistic regression probe (under 10 KB) on `Qwen/Qwen3.5-0.8B`, we can detect missing context in less than 0.1ms of probe overhead and physically cut off `model.generate()` to return `None`.

---

## Why This Exists

When building deterministic data-extraction pipelines (like parsing order emails, invoices, or ERP tickets) with small local models (`0.5B` to `4B`), the biggest headache isn't general world knowledge—it is **context faithfulness**. 

If you ask a small model, *"What is the total amount charged?"* and the email doesn't actually contain a price, the softmax layer still forces the model to pick the next most likely token. It will happily grab a street number, an order ID, or a historical price from last year and present it as the answer.

Standard fixes have annoying trade-offs:
* **Prompt engineering** ("If you don't know, say NULL") is brittle on sub-1B models.
* **External guardrails / LLM-as-a-Judge** require waiting for the model to finish hallucinating tokens, and then paying the latency and compute cost of a second forward pass to catch the lie.

**The question behind this experiment:** Since a Transformer's self-attention heads already cross-reference the question against the context during the initial prefill pass, does the model internally "know" that the target entity is missing before it opens its mouth? If so, can we just read that internal state and pull the plug early?

---

## How It Works (Read-Only Probing + Hard Cutoff)

We don't fine-tune the LLM, and we don't inject steering vectors into the residual stream (which can easily destabilize syntax in small models). We treat the LLM's prefill stage as a feature extractor and put a hard circuit breaker in front of the decoding loop:

```text
[Input: Order Email + Extraction Question]
                    |
                    v
     [LLM Prefill Pass (No Decoding)]
                    |
                    +---> (Tap Layer 15 Hidden State at Last Token: R^1024)
                                           |
                                           v
                          [Linear Probe: Risk = Sigmoid(W * h + b)]
                                           |
                        +------------------+------------------+
                        |                                     |
                 Risk < Threshold                      Risk >= Threshold
               (Target Span Found)                   (Target Span Missing)
                        |                                     |
                        v                                     v
           [Run model.generate()]              [Abort Decoding -> Return None]
