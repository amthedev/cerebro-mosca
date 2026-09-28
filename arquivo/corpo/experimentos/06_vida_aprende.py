"""Vida 2.0: a mosca aprende com a experiência e usa a memória depois.

Protocolo (o mesmo cérebro passa pelas três sessões):
  1. comida com cheiro A          -> aprende "A = açúcar"   (dopamina PAM)
  2. cheiro B + ataques de predador -> aprende "B = perigo" (dopamina PPL1)
  3. teste: fonte de A de um lado, fonte de B do outro, sem comida
Controle: uma mosca ingênua (sem treino) faz só o teste.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mosca.cerebro import RAIZ
from mosca.corpo import Comida, FonteCheiro, Predador
from mosca.video import montar_sessoes
from mosca.vida import Vida

p = argparse.ArgumentParser()
p.add_argument("--teste", type=float, default=12.0, help="duração do teste (s)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--repeticoes", type=int, default=0, help="moscas por grupo (a estatística completa está em 07_avaliar_memoria.py)")
args = p.parse_args()
pasta = RAIZ / "resultados"
pasta.mkdir(exist_ok=True)

FONTE_A = (16.0, 9.0)
FONTE_B = (16.0, -9.0)


def teste(vida: Vida, nome: str):
    vida.meta.energia = 0.5
    vida.novo_mundo(
        [], fontes=[FonteCheiro(*FONTE_A, "A"), FonteCheiro(*FONTE_B, "B")], nome=nome,
    )  # fmt: skip
    vida.viver(args.teste)
    return pd.DataFrame(vida.registro)


print("=== mosca treinada ===")
vida = Vida([Comida(14.0, 6.0, odores=("C", "A"))], memoria=True, energia=0.35, seed=args.seed,
            nome_sessao="1. Comida com cheiro A")  # fmt: skip
vida.viver(9.0)
vida.novo_mundo(
    [], [Predador(inicio_s=t, lado=lado) for t, lado in [(1.0, "left"), (2.6, "right"), (4.2, "left")]],
    cheiro_ambiente={"B": 0.8}, nome="2. Cheiro B + predador",
)  # fmt: skip
vida.viver(5.5)
treinada = teste(vida, "3. Teste: A ou B?")

pesos_treinados = vida.cerebro.pesos.copy()


def resumo(df):
    da = np.hypot(df.x - FONTE_A[0], df.y - FONTE_A[1])
    db = np.hypot(df.x - FONTE_B[0], df.y - FONTE_B[1])
    perto_a, perto_b = float((da < 6).mean()), float((db < 6).mean())
    indice = (perto_a - perto_b) / max(perto_a + perto_b, 1e-9)  # +1 só A, -1 só B
    return perto_a, perto_b, indice


print("=== estatística: moscas treinadas x ingênuas ===")
aval = Vida([], memoria=True, energia=0.5, renderizar=False, nome_sessao="avaliação") if args.repeticoes else None
pesos_ingenuos = aval.cerebro.pesos.copy() if aval else None
linhas, trajetos = [], {"treinada": [treinada], "ingênua": []}
for k in range(args.repeticoes):
    for grupo, pesos in [("treinada", pesos_treinados), ("ingênua", pesos_ingenuos)]:
        aval.cerebro.pesos[:] = pesos
        aval.cerebro.reiniciar(seed=100 + k)
        aval.seed = 100 + k
        df = teste(aval, grupo)
        a, b, ind = resumo(df)
        linhas.append(dict(grupo=grupo, seed=100 + k, perto_A=a, perto_B=b, indice=ind))
        trajetos[grupo].append(df)
        print(f"  {grupo:9s} seed {100 + k}: perto de A {a:5.1%}  perto de B {b:5.1%}  índice {ind:+.2f}")
est = pd.DataFrame(linhas, columns=["grupo", "seed", "perto_A", "perto_B", "indice"])
if len(est):
    est.to_csv(pasta / "vida_aprende_estatistica.csv", index=False)
    print(est.groupby("grupo")[["perto_A", "perto_B", "indice"]].mean().round(3).to_string())

fig, ax = plt.subplots(figsize=(6.5, 5.5))
for (x, y), cor, nome in [(FONTE_A, "#3366ff", "cheiro A (era comida)"), (FONTE_B, "#b040c0", "cheiro B (era perigo)")]:
    ax.add_patch(plt.Circle((x, y), 6, color=cor, alpha=0.15))
    ax.plot(x, y, "o", color=cor, ms=10, label=nome)
for grupo, cor in [("ingênua", "#999999"), ("treinada", "#1a8f5a")]:
    for i, df in enumerate(trajetos[grupo]):
        ax.plot(df.x, df.y, color=cor, lw=1.6, alpha=0.8, label=f"moscas {grupo}s" if i == 0 else None)
ax.plot(0, 0, "k^", ms=9, label="início")
ax.set_aspect("equal")
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
media = est.groupby("grupo").indice.mean()
ax.set_title(f"Teste de memória (índice de preferência por A)\ntreinadas {media.get('treinada', 0):+.2f}   ingênuas {media.get('ingênua', 0):+.2f}")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig(pasta / "vida_aprende_trajetorias.png", dpi=150)

treinada.to_csv(pasta / "vida_aprende_teste.csv", index=False)
legendas = {
    "1. Comida com cheiro A": "a mosca com fome acha comida que tem cheiro A\nenquanto come, a dopamina de recompensa marca o cheiro A",
    "2. Cheiro B + predador": "o ambiente tem cheiro B e um predador ataca três vezes\na dopamina de punição marca o cheiro B",
    "3. Teste: A ou B?": "sem comida e sem predador: só os dois cheiros\na mosca usa o que lembra",
}
video = montar_sessoes(vida.sessoes, pasta / "vida_aprende.mp4", legendas=legendas)
print("vídeo:", video)
