"""Quais neurônios descendentes (cérebro -> corpo) respondem a cada sentido?

Estimula grupos sensoriais de um lado de cada vez e mede todos os ~1300
neurônios descendentes. Serve para desenhar a ponte cérebro -> corpo com base
no que o próprio conectoma faz, em vez de só escolher neurônios à mão.
"""

import time

import numpy as np
import pandas as pd

from mosca.cerebro import PASTA_DADOS, Cerebro
from mosca.neuronios import Catalogo

TAXA_HZ = 100
DURACAO_MS = 500
TENTATIVAS = 2
CANDIDATOS = ["MDN", "DNp09", "DNa01", "DNa02", "DNp01", "DNg12_a", "CB0701"]

cerebro = Cerebro(seed=0)
cat = Catalogo(cerebro.con.ids)

estimulos = {}
for lado in ("left", "right"):
    L = lado[0].upper()
    estimulos[f"acucar_{L}"] = cat.acucar(lado)
    estimulos[f"amargo_{L}"] = cat.amargo(lado)
    estimulos[f"cheiro_comida_{L}"] = cat.olfato_comida(lado)
    estimulos[f"cheiro_perigo_{L}"] = cat.olfato_perigo(lado)
    estimulos[f"looming_{L}"] = cat.lplc2(lado)
    estimulos[f"antena_JO_{L}"] = cat.orgao_johnston(lado)

saidas = cat.descendentes + cat.mn9()
idx_saidas = cerebro.idx(saidas)
resultados = {}
for nome, ids in estimulos.items():
    t = time.time()
    cerebro.taxa_estimulo[:] = 0
    cerebro.estimular(ids, TAXA_HZ)
    taxas = np.zeros(len(idx_saidas))
    for k in range(TENTATIVAS):
        cerebro.reiniciar(seed=k)
        cerebro.zerar_contagem()
        cerebro.rodar(DURACAO_MS)
        taxas += cerebro.taxas()[idx_saidas]
    resultados[nome] = taxas / TENTATIVAS
    print(f"{nome:18s} {len(ids):4d} neurônios estimulados  ({time.time() - t:.0f} s)")

tabela = pd.DataFrame(resultados, index=[cat.nome(i) for i in saidas])
tabela.to_csv(PASTA_DADOS / "mapa_saidas.csv")

print("\n=== descendentes mais ativos por estímulo (Hz) ===")
for nome in tabela.columns:
    top = tabela[nome].sort_values(ascending=False).head(8)
    top = top[top > 0.5]
    print(f"{nome:18s} " + ", ".join(f"{n}={v:.0f}" for n, v in top.items()))

print("\n=== neurônios candidatos para comandar o corpo (Hz) ===")
cand = tabela[[any(n.startswith(c + "_") for c in CANDIDATOS) for n in tabela.index]]
print(cand.round(1).to_string())
