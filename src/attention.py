"""Utilidades para la HT2: atención implementada desde cero y análisis de
atenciones de modelos preentrenados (BERT / GPT-2) de Hugging Face."""
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Implementación desde cero
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(Q, K, V, mask=None, scale=True):
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.

    Q: (..., n_q, d_k), K: (..., n_k, d_k), V: (..., n_k, d_v).
    mask: tensor booleano broadcastable a (..., n_q, n_k); True = posición permitida.
    Devuelve (salida, pesos de atención).
    """
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1)
    if scale:
        scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ V, weights


def causal_mask(n, device=None):
    """Máscara triangular inferior (True = el token i puede ver al token j <= i)."""
    return torch.tril(torch.ones(n, n, dtype=torch.bool, device=device))


def multi_head_attention(x, W_q, W_k, W_v, W_o, num_heads, mask=None):
    """Multi-head self-attention con pesos dados (convención de nn.Linear: y = x W^T).

    x: (batch, n, d_model). Divide d_model en h cabezas de tamaño d_model / h,
    aplica atención en paralelo y concatena. Devuelve (salida, pesos (batch, h, n, n)).
    """
    B, n, d_model = x.shape
    d_head = d_model // num_heads

    def split(t):  # (B, n, d_model) -> (B, h, n, d_head)
        return t.view(B, n, num_heads, d_head).transpose(1, 2)

    Q, K, V = split(x @ W_q.T), split(x @ W_k.T), split(x @ W_v.T)
    out, w = scaled_dot_product_attention(Q, K, V, mask=mask)
    out = out.transpose(1, 2).reshape(B, n, d_model)
    return out @ W_o.T, w


# ---------------------------------------------------------------------------
# Modelos preentrenados
# ---------------------------------------------------------------------------
def get_attentions(model, tokenizer, text):
    """Corre el modelo y devuelve (tokens, atenciones) con atenciones de forma
    (capas, cabezas, n, n) para un único ejemplo."""
    enc = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    tokens = [t.replace("Ġ", "") for t in tokens]  # quita el marcador de espacio de GPT-2
    return tokens, torch.stack(out.attentions)[:, 0]


def attention_entropy(A, normalize=False):
    """Entropía (nats) de cada fila de atención. A: (..., n, n).

    Con normalize=True divide entre la entropía máxima posible de cada fila
    (log del número de posiciones visibles), lo que permite comparar BERT
    (ve n tokens) con GPT-2 (la fila i solo ve i+1 tokens)."""
    H = -(A * torch.log(A.clamp_min(1e-12))).sum(-1)
    if normalize:
        visible = (A > 0).sum(-1).clamp_min(2).float()
        H = H / torch.log(visible)
    return H


def head_patterns(A):
    """Para cada cabeza, atención promedio al token anterior, siguiente, a sí mismo,
    al primer token ([CLS]) y al último ([SEP]). A: (L, H, n, n)."""
    return {
        "anterior": A.diagonal(-1, dim1=-2, dim2=-1).mean(-1),
        "siguiente": A.diagonal(1, dim1=-2, dim2=-1).mean(-1),
        "misma": A.diagonal(0, dim1=-2, dim2=-1).mean(-1),
        "primero": A[..., :, 0].mean(-1),
        "último": A[..., :, -1].mean(-1),
    }


def plot_heads(A, tokens, heads, title=None, ncols=4, cmap="viridis"):
    """Mapas de calor de varias cabezas. heads: lista de (capa, cabeza, etiqueta)."""
    nrows = math.ceil(len(heads) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows))
    for ax, (l, h, label) in zip(np.atleast_1d(axes).flat, heads):
        ax.imshow(A[l, h].numpy(), cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(tokens)), tokens, rotation=90, fontsize=7)
        ax.set_yticks(range(len(tokens)), tokens, fontsize=7)
        ax.set_title(f"Capa {l}, cabeza {h}\n{label}", fontsize=9)
    for ax in np.atleast_1d(axes).flat[len(heads):]:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
