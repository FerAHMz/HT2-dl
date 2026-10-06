# HT2 — Transformers y mecanismos de atención

CC3092 Deep Learning y Sistemas Inteligentes · Fernando Andree Hernández Martínez (23645)

Implementación de *scaled dot-product attention* y *multi-head attention* desde cero, verificación contra PyTorch y análisis de los pesos de atención de BERT (`bert-base-uncased`) y GPT-2 (`gpt2`).

## Estructura

```
HT2-dl/
├── src/attention.py           # atención desde cero + utilidades de análisis y gráficas
├── notebooks/HT2_atencion.ipynb  # notebook completo, ejecutado y comentado
├── figures/                   # figuras exportadas por el notebook (usadas en el PDF)
├── requirements.txt
└── README.md
```

No hay dataset: las entradas son oraciones escritas en el notebook y los modelos se descargan de Hugging Face la primera vez (~1 GB).

## Entorno y ejecución

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/HT2_atencion.ipynb
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/HT2_atencion.ipynb
```

También se puede abrir con `jupyter lab` y correr las celdas en orden. Las semillas se fijan en la primera celda.

## Contenido del notebook

1. Configuración y semillas.
2. Ejercicio a mano de atención (3 tokens, $d_k=2$) y verificación numérica.
3. Verificaciones: implementación propia vs `F.scaled_dot_product_attention`, `nn.MultiheadAttention`, máscara causal y `generate_square_subsequent_mask`.
4. Escalamiento por $\sqrt{d_k}$: varianza de $q\cdot k$, saturación de la softmax y gradiente.
5. Costo en tiempo y memoria de la self-attention respecto a $n$.
6. Mapas de calor de cabezas especializadas en BERT y GPT-2.
7. Correferencia de «it» (*too tired* vs *too wide*) en BERT.
8. GPT-2: matriz triangular inferior y *attention sinks*.
9. Entropía de la atención por capa (cruda y normalizada).
10. (Opcional) bertviz.
