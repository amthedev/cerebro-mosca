"""Quão perto da explosão o modelo trabalha? Varredura do peso por sinapse.

O modelo tem um único peso por sinapse (0,275 mV, escolhido no artigo
original). Aqui ele é multiplicado por um fator nas duas moscas, com as
correções ligadas, e medimos: quantos neurônios disparam, as saídas-chave
(MN9 com açúcar, Fibra Gigante com LPLC2, aBN1/SAD093 com a antena) e a
concordância com o FlyWire no peso original (Spearman por tipo, como no 13).
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mosca import banc, biologia
from mosca.cerebro import RAIZ, Cerebro, Parametros, carregar_conectoma
from mosca.neuronios import Catalogo

PROTOCOLOS = {
    "açúcar": (lambda cat: cat.ids("left", cell_type="LB3"), "CB0701"),  # MN9
    "ameaça": (lambda cat: cat.lplc2("left"), "DNp01"),  # Fibra Gigante
    "antena": (lambda cat: cat.ids("left", cell_type=lambda t: t.startswith(("JO-C", "JO-E"))), "SAD093"),  # aBN1
}
FATORES = {"FlyWire": (0.7, 0.85, 1.0, 1.15, 1.3, 1.5), "BANC": (1.0, 1.2, 1.4, 1.6, 1.8, 2.12)}


def por_tipo(cat, con, tx):
    ann = cat.ann.reindex(con.ids)
    return pd.Series(tx).groupby([ann.cell_type.to_numpy(), ann.side.fillna("?").to_numpy()]).mean()


fw_con = carregar_conectoma()
b_meta = banc.metadados()
b_con = banc.carregar_cerebro(b_meta)
moscas = {"FlyWire": (fw_con, Catalogo(fw_con.ids)), "BANC": (b_con, banc.CatalogoBANC(b_con.ids, b_meta))}

referencia, linhas = {}, []
for nome in ("FlyWire", "BANC"):
    con, cat = moscas[nome]
    for f in FATORES[nome]:
        c = Cerebro(con, params=Parametros(w_syn=0.275 * f), seed=0)
        biologia.aplicar(c, cat)
        for prot, (escolher, saida) in PROTOCOLOS.items():
            c.taxa_estimulo[:] = 0
            c.estimular_idx(c.idx(escolher(cat)), 150)
            c.reiniciar(0)
            c.zerar_contagem()
            c.rodar(1000)
            tx = c.taxas()
            tipos = por_tipo(cat, con, tx)
            if nome == "FlyWire" and f == 1.0:
                referencia[prot] = tipos
            rho = np.nan
            if prot in referencia:
                ref = referencia[prot]
                comum = ref.index.intersection(tipos.index)
                estim = set(cat.ann.loc[escolher(cat), "cell_type"])
                comum = comum[~comum.get_level_values(0).isin(estim)]
                x, y = ref[comum].to_numpy(), tipos[comum].to_numpy()
                resp = (x > 2) | (y > 2)
                rho = spearmanr(x[resp], y[resp]).statistic
            s = tx[c.idx(cat.ids(cell_type=saida))].mean() if saida != "CB0701" else tx[c.idx(cat.mn9())].mean()
            linhas.append({"mosca": nome, "fator": f, "protocolo": prot, "neurônios ativos": int((tx > 2).sum()),
                           "saída (Hz)": s, "Spearman x FlyWire 1,0": rho})  # fmt: skip
            print(f"{nome:8s} x{f:<5} {prot:7s} ativos {int((tx > 2).sum()):6d} | saída {s:6.1f} Hz | ρ {rho:5.2f}", flush=True)

pd.DataFrame(linhas).to_csv(RAIZ / "resultados" / "escala.csv", index=False)
