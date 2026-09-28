"""Avaliação estatística da memória: moscas treinadas x ingênuas no teste A/B.

O treino (comida com cheiro A; cheiro B + predador) roda uma vez e os pesos do
cérebro treinado ficam salvos em dados/pesos_treinados.npy. Depois cada mosca
(semente diferente) faz o teste sem vídeo.

    uv run python experimentos/07_avaliar_memoria.py --grupo treinada --n 5
    uv run python experimentos/07_avaliar_memoria.py --grupo ingenua --n 5
"""

import argparse
import json

import numpy as np
import pandas as pd

from mosca.cerebro import PASTA_DADOS, RAIZ
from mosca.corpo import Comida, FonteCheiro, Predador
from mosca.vida import Vida

p = argparse.ArgumentParser()
p.add_argument("--grupo", choices=["treinada", "ingenua"], required=True)
p.add_argument("--n", type=int, default=5)
p.add_argument("--teste", type=float, default=12.0)
p.add_argument("--retreinar", action="store_true")
args = p.parse_args()

FONTE_A, FONTE_B = (16.0, 9.0), (16.0, -9.0)
ARQ = PASTA_DADOS / "pesos_treinados.npy"


def treinar() -> np.ndarray:
    vida = Vida([Comida(14.0, 6.0, odores=("C", "A"))], memoria=True, energia=0.35, renderizar=False)
    vida.viver(9.0, mostrar_cada_s=3)
    vida.novo_mundo(
        [], [Predador(inicio_s=t, lado=lado) for t, lado in [(1.0, "left"), (2.6, "right"), (4.2, "left")]],
        cheiro_ambiente={"B": 0.8},
    )  # fmt: skip
    vida.viver(5.5, mostrar_cada_s=3)
    np.save(ARQ, vida.cerebro.pesos)
    return vida.cerebro.pesos.copy()


vida = Vida([], memoria=True, energia=0.5, renderizar=False)
if args.grupo == "treinada":
    pesos = np.load(ARQ) if ARQ.exists() and not args.retreinar else treinar()
    vida.cerebro.pesos[:] = pesos
resultados = []
for k in range(args.n):
    seed = 100 + k
    vida.cerebro.reiniciar(seed=seed)
    vida.seed = seed
    vida.meta.energia = 0.5
    vida.novo_mundo([], fontes=[FonteCheiro(*FONTE_A, "A"), FonteCheiro(*FONTE_B, "B")])
    vida.viver(args.teste, mostrar_cada_s=0)
    df = pd.DataFrame(vida.registro)
    da = np.hypot(df.x - FONTE_A[0], df.y - FONTE_A[1])
    db = np.hypot(df.x - FONTE_B[0], df.y - FONTE_B[1])
    a, b = float((da < 6).mean()), float((db < 6).mean())
    ind = (a - b) / max(a + b, 1e-9)
    resultados.append(dict(grupo=args.grupo, seed=seed, perto_A=a, perto_B=b, indice=ind,
                           x=df.x.round(2).tolist(), y=df.y.round(2).tolist()))  # fmt: skip
    print(f"{args.grupo} seed {seed}: perto de A {a:5.1%}  perto de B {b:5.1%}  índice {ind:+.2f}", flush=True)
(RAIZ / "resultados").mkdir(exist_ok=True)
with open(RAIZ / "resultados" / f"memoria_{args.grupo}.json", "w") as f:
    json.dump(resultados, f)
print(f"MÉDIA {args.grupo}: índice {np.mean([r['indice'] for r in resultados]):+.2f}")
