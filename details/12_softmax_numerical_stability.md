# The Softmax Numerical Overflow / Underflow Explosion

To prevent floating-point representation limits from overflowing when calculating exponentials, compilers apply the Log-Sum-Exp identity.

## Mathematical Formulation
$$\ln \sum \exp(z_i) = z_{\text{max}} + \ln \sum \exp(z_i - z_{\text{max}})$$

## Diagram
```mermaid
flowchart TD
    Logits[Raw Logits] --> Max[Find Max Logit]
    Max --> Sub[Subtract Max from Logits]
    Sub --> Exp[Safe Exponentiation]
    Exp --> Sum[Sum and Log-Sum-Exp]
```

[Back to README](../README.md)
