"""Gráfico das trajetórias do teste de memória (saída de 07_avaliar_memoria.py)."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mosca.cerebro import RAIZ

pasta = RAIZ / "resultados"
FONTE_A, FONTE_B = (16.0, 9.0), (16.0, -9.0)
grupos = {g: json.load(open(pasta / f"memoria_{g}.json")) for g in ("ingenua", "treinada")}

fig, eixos = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
for ax, (g, titulo, cor) in zip(eixos, [("ingenua", "Moscas ingênuas (sem treino)", "#888888"),
                                        ("treinada", "Moscas treinadas (A = comida, B = perigo)", "#1a8f5a")]):  # fmt: skip
    for (x, y), c, nome in [(FONTE_A, "#3366ff", "cheiro A"), (FONTE_B, "#b040c0", "cheiro B")]:
        ax.add_patch(plt.Circle((x, y), 6, color=c, alpha=0.15))
        ax.plot(x, y, "o", color=c, ms=10, label=nome)
    for r in grupos[g]:
        ax.plot(r["x"], r["y"], color=cor, lw=1.4, alpha=0.85)
    ax.plot(0, 0, "k^", ms=9)
    ind = np.mean([r["indice"] for r in grupos[g]])
    ax.set_title(f"{titulo}\níndice de preferência por A: {ind:+.2f} (n={len(grupos[g])})", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
eixos[0].set_ylabel("y (mm)")
eixos[0].legend(loc="lower left", fontsize=8)
fig.suptitle("Teste de memória: fonte de A e fonte de B, sem comida e sem predador")
fig.tight_layout()
fig.savefig(pasta / "memoria_trajetorias.png", dpi=140)
print(pasta / "memoria_trajetorias.png")
