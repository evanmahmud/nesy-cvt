# Mathematics

The full training objective is:

```text
L = L_cls + lambda_rule L_rule + lambda_concept L_concept
    + lambda_proto L_proto + lambda_spectral L_spectral
```

## Classification Loss

`L_cls` is standard cross entropy over class logits.

## Complex Spectral Mixing

Given feature map `x`, the mixer computes:

```text
X = FFT2(x)
G = tanh(A) + i tanh(B)
Y = X * (1 + G)
y = IFFT2(Y)
```

`A` and `B` are learnable real tensors. The hyperbolic tangent bounds the complex gate. The corrected default returns `y`, so zero gates give `y=x`. Historical checkpoints used `x+y`, which doubles the feature map at zero gates; their config-less spectral behavior is preserved under `spectral_mixer_mode=legacy_double`.

## Vector Prototype Classification

For feature vector `z` and class prototype `p_c`:

```text
logit_c = s * cosine(z, p_c)
```

The prototype loss pulls each feature toward its label prototype:

```text
L_proto = 1 - cosine(z, p_y)
```

## Learned Auxiliary Concept-to-Class Loss

The concept head predicts concept logits `c`. Fuzzy concepts are:

```text
q = sigmoid(c)
```

A learned linear projection maps concepts to auxiliary class logits:

```text
r = W_rule q
```

The implementation calls this `rule_loss` for historical reasons, but cross entropy through a learned linear projection is not an explicit symbolic rule system. CIFAR-100 training does not provide concept targets; the optional concept loss is therefore absent in current runs. A future neuro-symbolic claim needs semantic concept targets, explicit rules or a justified logic operator, and rule-level evaluation.

## Numerical Analysis Notes

- FFT uses orthonormal normalization to reduce scale drift.
- Complex gates are bounded with `tanh`.
- Residual paths preserve spatial features if spectral mixing is initially unhelpful.
- Training should use gradient clipping and NaN checks.
