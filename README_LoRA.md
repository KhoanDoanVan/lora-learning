# LoRA: Low-Rank Adaptation

LoRA is a parameter-efficient fine-tuning technique. Instead of updating all pretrained weights of a large neural network, LoRA freezes the original weights and trains a small low-rank update matrix.

paper: https://arxiv.org/abs/2106.09685

origin source for coding: https://github.com/sunildkumar/lora_from_scratch

---

## 1. Standard Linear Layer

Consider a pretrained linear layer:

```math
h = W_0 x
```

where:

- \(x \in \mathbb{R}^{d_{\text{in}}}\) is the input vector.
- \(W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\) is the pretrained weight matrix.
- \(h \in \mathbb{R}^{d_{\text{out}}}\) is the output vector.

In full fine-tuning, we update the entire matrix \(W_0\). That means the trainable parameter count is:

```math
d_{\text{out}} \times d_{\text{in}}
```

For large language models, this is extremely expensive.

---

## 2. Core Idea of LoRA

LoRA assumes that the update needed for downstream adaptation does not need to be full-rank.

Instead of learning a dense update matrix \(\Delta W\), LoRA represents it as a product of two smaller matrices:

```math
\Delta W = B A
```

where:

```math
A \in \mathbb{R}^{r \times d_{\text{in}}}
```

```math
B \in \mathbb{R}^{d_{\text{out}} \times r}
```

and \(r\) is the **rank** of the LoRA adapter.

The adapted layer becomes:

```math
h = W_0 x + \Delta W x
```

Substituting the low-rank decomposition:

```math
h = W_0 x + B A x
```

In practice, LoRA usually applies a scaling factor:

```math
h = W_0 x + \frac{\alpha}{r} B A x
```

where:

- \(\alpha\) is the LoRA scaling hyperparameter.
- \(r\) is the LoRA rank.
- \(\frac{\alpha}{r}\) stabilizes the magnitude of the LoRA update.

---

## 3. Shape Intuition

For a square weight matrix:

```math
W_0 \in \mathbb{R}^{d \times d}
```

LoRA uses:

```math
A \in \mathbb{R}^{r \times d}
```

```math
B \in \mathbb{R}^{d \times r}
```

The computation is:

```math
x \in \mathbb{R}^{d}
```

```math
A x \in \mathbb{R}^{r}
```

```math
B(Ax) \in \mathbb{R}^{d}
```

So the LoRA branch maps:

```math
\mathbb{R}^{d} \rightarrow \mathbb{R}^{r} \rightarrow \mathbb{R}^{d}
```

This is a bottleneck structure.

---

## 4. Why Is It Called Low-Rank?

The rank of a matrix product is bounded by the smallest internal dimension:

```math
\operatorname{rank}(BA) \leq \min(\operatorname{rank}(B), \operatorname{rank}(A)) \leq r
```

Therefore:

```math
\operatorname{rank}(\Delta W) \leq r
```

This means LoRA restricts the weight update to a low-rank subspace.

If \(r \ll d\), then LoRA learns a much smaller update than full fine-tuning.

---

## 5. Parameter Efficiency

Full fine-tuning trains:

```math
d_{\text{out}} d_{\text{in}}
```

parameters.

LoRA trains only:

```math
r d_{\text{in}} + d_{\text{out}} r
```

parameters.

So the number of trainable LoRA parameters is:

```math
r(d_{\text{in}} + d_{\text{out}})
```

For a square matrix \(d \times d\):

Full fine-tuning:

```math
d^2
```

LoRA fine-tuning:

```math
2dr
```

The compression ratio is approximately:

```math
\frac{2dr}{d^2} = \frac{2r}{d}
```

Example:

If \(d = 4096\) and \(r = 8\):

```math
\text{LoRA parameters} = 2 \times 4096 \times 8 = 65536
```

Full fine-tuning parameters:

```math
4096^2 = 16777216
```

Ratio:

```math
\frac{65536}{16777216} \approx 0.0039
```

So LoRA trains only about:

```math
0.39\%
```

of the parameters of that layer.

---

## 6. Initialization

A common LoRA initialization is:

```math
A \sim \mathcal{N}(0, \sigma^2)
```

```math
B = 0
```

Therefore, at the beginning of training:

```math
\Delta W = BA = 0
```

So the initial model behaves exactly like the original pretrained model:

```math
h = W_0 x
```

This is important because LoRA does not disturb the pretrained model at step zero.

During training, \(B\) starts learning first because \(A\) is already random. Once \(B\) becomes non-zero, \(A\) also receives meaningful gradients.

---

## 7. Training Objective

Given a dataset:

```math
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}
```

LoRA fine-tuning solves:

```math
\min_{A,B} \sum_{i=1}^{N} \mathcal{L}
\left(
f(x_i; W_0, A, B), y_i
\right)
```

subject to:

```math
W_0 \text{ is frozen}
```

Only \(A\) and \(B\) are updated.

The pretrained weight \(W_0\) does not receive gradients:

```math
\nabla_{W_0}\mathcal{L} = 0
```

The LoRA matrices receive gradients:

```math
\nabla_A \mathcal{L} \neq 0
```

```math
\nabla_B \mathcal{L} \neq 0
```

---

## 8. Rank \(r\): What Does It Control?

The rank \(r\) controls the capacity of the LoRA adapter.

A small rank means:

- Fewer trainable parameters.
- Lower memory usage.
- Lower risk of overfitting.
- Less adaptation capacity.

A large rank means:

- More trainable parameters.
- Higher memory usage.
- Greater adaptation capacity.
- Potentially better performance on complex tasks.

The rank is a hyperparameter.

LoRA itself does **not** automatically search for the best rank like Neural Architecture Search. In standard LoRA, the practitioner chooses \(r\), often by validation experiments.

For example:

```text
r = 4   -> very lightweight adaptation
r = 8   -> common default for many tasks
r = 16  -> stronger adaptation
r = 32+ -> more capacity, higher cost
```

There are adaptive variants, such as AdaLoRA, that try to allocate rank more dynamically, but standard LoRA does not perform rank search by itself.

---

## 9. LoRA in Transformer Models

In Transformer-based language models, LoRA is usually inserted into linear projection matrices, such as:

- Query projection: \(W_q\)
- Key projection: \(W_k\)
- Value projection: \(W_v\)
- Output projection: \(W_o\)
- MLP up-projection
- MLP down-projection
- MLP gate projection

For example, applying LoRA to the query projection:

```math
q = W_q x
```

becomes:

```math
q = W_q x + \frac{\alpha}{r} B_q A_q x
```

where:

```math
A_q \in \mathbb{R}^{r \times d_{\text{model}}}
```

```math
B_q \in \mathbb{R}^{d_{\text{model}} \times r}
```

---

## 10. Inference-Time Merging

After LoRA training, the adapter can be merged into the original weight:

```math
W_{\text{merged}} = W_0 + \frac{\alpha}{r}BA
```

Then inference becomes the standard linear operation:

```math
h = W_{\text{merged}}x
```

This means LoRA can add no extra inference latency after merging.

---

## 11. PyTorch-Style Implementation

A simplified LoRA linear layer can be written as:

```python
import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float):
        super().__init__()

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Freeze pretrained weight
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # A: down projection
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))

        # B: up projection
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))

        # Initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_output = self.base_layer(x)

        # x shape: [batch_size, in_features]
        # LoRA update: x @ A.T @ B.T
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

        return base_output + lora_output
```

The mathematical equivalent is:

```math
\text{output} = xW_0^T + \frac{\alpha}{r}xA^TB^T
```

This is the batch-form version of:

```math
h = W_0 x + \frac{\alpha}{r}BAx
```

---

## 12. Why LoRA Works Well

LoRA works well because many downstream adaptations appear to lie in a low-dimensional subspace.

Instead of changing every direction in the pretrained weight space, LoRA learns a small number of useful update directions.

Conceptually:

```math
W_{\text{task}} = W_0 + \Delta W
```

LoRA approximates:

```math
\Delta W \approx BA
```

with:

```math
\operatorname{rank}(\Delta W) \leq r
```

This makes fine-tuning cheaper while preserving much of the expressive power needed for adaptation.

---

## 13. LoRA vs Full Fine-Tuning

| Aspect | Full Fine-Tuning | LoRA |
|---|---:|---:|
| Pretrained weights updated? | Yes | No |
| Trainable parameters | Very high | Low |
| GPU memory usage | High | Much lower |
| Storage per task | Full model checkpoint | Small adapter |
| Risk of catastrophic forgetting | Higher | Lower |
| Inference overhead | None | None if merged |
| Best for | Maximum flexibility | Efficient adaptation |

---

## 14. Practical Notes

Useful LoRA hyperparameters include:

| Hyperparameter | Meaning |
|---|---|
| \(r\) | Rank of the low-rank adapter |
| \(\alpha\) | Scaling factor |
| LoRA dropout | Dropout applied to the LoRA branch |
| Target modules | Which layers receive LoRA adapters |
| Learning rate | Usually larger than full fine-tuning learning rates |
| Bias training | Whether to train biases or keep them frozen |

A common starting point:

```text
rank r = 8 or 16
alpha = 16 or 32
dropout = 0.05
target modules = q_proj, v_proj
```

For stronger adaptation, LoRA is often applied to more modules, such as:

```text
q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj
```

---

## 15. Summary

LoRA modifies a pretrained layer:

```math
h = W_0x
```

into:

```math
h = W_0x + \frac{\alpha}{r}BAx
```

where:

```math
A \in \mathbb{R}^{r \times d_{\text{in}}}
```

```math
B \in \mathbb{R}^{d_{\text{out}} \times r}
```

and:

```math
\operatorname{rank}(BA) \leq r
```

The original pretrained weight \(W_0\) is frozen, and only \(A\) and \(B\) are trained.

The main benefit is that LoRA provides efficient task adaptation with far fewer trainable parameters than full fine-tuning.
