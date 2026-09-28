"""Mapa descendentes -> pernas pela medula real (conectoma BANC).

Estimula cada tipo de neurônio descendente (de um lado) e mede os
motoneurônios de cada perna, separados pela função (flexionar/estender cada
articulação). Testes de validação com circuitos conhecidos:
  - Fibra Gigante (DNp01) -> motoneurônio de salto (TTM) das pernas do meio
  - MDN (andar para trás) e DNp09 (andar para frente)
  - DNa01/DNa02 de um lado (virar): assimetria entre pernas esquerdas e direitas
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mosca.cerebro import RAIZ
from mosca.medula import Medula

TAXA = 150.0
med = Medula()
ORDEM_PERNAS = ["lf", "lm", "lh", "rf", "rm", "rh"]
TESTES = [
    ("DNp01", "left"), ("DNp01", "right"), ("MDN", None), ("DNp09", None),
    ("DNa01", "left"), ("DNa01", "right"), ("DNa02", "left"), ("DNa02", "right"),
    ("DNg12_a", None), ("DNb05", None),
]  # fmt: skip

linhas, por_funcao = [], {}
for tipo, lado in TESTES:
    idx = med.dns(tipo, lado)
    if len(idx) == 0:
        print(f"{tipo} {lado}: não encontrado")
        continue
    rs = [med.resposta([(idx, TAXA)], seed=s) for s in range(2)]
    df = pd.concat(rs).groupby(["perna", "funcao"], as_index=False).hz.mean()
    rotulo = f"{tipo}" + (f" {lado[0].upper()}" if lado else " (ambos)")
    por_perna = df.groupby("perna").hz.sum().reindex(ORDEM_PERNAS).fillna(0)  # soma dos motoneurônios da perna
    salto = np.mean([r for r in [med.c.taxas()[med.salto].mean()]])
    linhas.append({"DN": rotulo, **por_perna.to_dict(), "salto_TTM": salto})
    por_funcao[rotulo] = df.pivot_table(index="funcao", columns="perna", values="hz").reindex(columns=ORDEM_PERNAS)
    print(f"{rotulo:14s} " + "  ".join(f"{p}={por_perna[p]:5.1f}" for p in ORDEM_PERNAS) + f"  | salto(TTM)={salto:5.1f} Hz", flush=True)

tab = pd.DataFrame(linhas).set_index("DN")
tab.to_csv(RAIZ / "resultados" / "medula_mapa.csv")

fig, eixos = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.4]})
im = eixos[0].imshow(tab[ORDEM_PERNAS].values, cmap="viridis", aspect="auto")
eixos[0].set_xticks(range(6), ["esq.\nfrente", "esq.\nmeio", "esq.\ntrás", "dir.\nfrente", "dir.\nmeio", "dir.\ntrás"])
eixos[0].set_yticks(range(len(tab)), tab.index)
eixos[0].set_title("Atividade total dos motoneurônios de cada perna (Hz)\nquando cada descendente é ativado (medula BANC)")
plt.colorbar(im, ax=eixos[0])
chave = "DNa02 L" if "DNa02 L" in por_funcao else list(por_funcao)[0]
f = por_funcao[chave].fillna(0)
im2 = eixos[1].imshow(f.values, cmap="magma", aspect="auto")
eixos[1].set_xticks(range(6), ORDEM_PERNAS)
eixos[1].set_yticks(range(len(f)), [s.replace("_", " ") for s in f.index], fontsize=7)
eixos[1].set_title(f"{chave}: por função do músculo")
plt.colorbar(im2, ax=eixos[1])
fig.tight_layout()
fig.savefig(RAIZ / "resultados" / "medula_mapa.png", dpi=130)
print("figura:", RAIZ / "resultados" / "medula_mapa.png")
