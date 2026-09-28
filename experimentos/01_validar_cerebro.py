"""Valida o motor numba contra o modelo original em Brian2 (experimento do açúcar).

Estimula os neurônios gustativos de açúcar (lista do tutorial de Shiu et al.)
e compara a taxa de disparo dos neurônios mais ativos nos dois motores,
incluindo o MN9, neurônio motor que estende a probóscide.
"""

import sys
import time

import numpy as np
import pandas as pd

from mosca.cerebro import PASTA_DADOS, PASTA_SHIU, Cerebro

N_TENTATIVAS = 10
DURACAO_MS = 1000
TAXA_HZ = 150
MN9 = 720575940660219265
ACUCAR = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]  # fmt: skip


def rodar_numba() -> pd.Series:
    cerebro = Cerebro(seed=0)
    acucar = [a for a in ACUCAR if a in cerebro.indice]
    cerebro.estimular(acucar, TAXA_HZ)
    taxas = np.zeros(cerebro.n)
    inicio = time.time()
    for k in range(N_TENTATIVAS):
        cerebro.reiniciar(seed=k)
        cerebro.zerar_contagem()
        cerebro.rodar(DURACAO_MS)
        taxas += cerebro.taxas()
    print(f"numba: {N_TENTATIVAS} x {DURACAO_MS} ms em {time.time() - inicio:.1f} s")
    return pd.Series(taxas / N_TENTATIVAS, index=cerebro.con.ids)


def rodar_brian2() -> pd.Series:
    sys.path.insert(0, str(PASTA_SHIU))
    from brian2 import Hz
    from model import default_params, run_exp

    import utils

    params = dict(default_params, n_run=N_TENTATIVAS, r_poi=TAXA_HZ * Hz)
    pasta = PASTA_DADOS / "validacao_brian2"
    pasta.mkdir(parents=True, exist_ok=True)
    comp = PASTA_SHIU / "Completeness_783.csv"
    ids = set(pd.read_csv(comp, index_col=0).index)
    inicio = time.time()
    run_exp(
        exp_name="acucar",
        neu_exc=[a for a in ACUCAR if a in ids],
        path_res=pasta,
        path_comp=comp,
        path_con=PASTA_SHIU / "Connectivity_783.parquet",
        params=params,
        n_proc=-1,
    )
    print(f"brian2: {time.time() - inicio:.1f} s (ou cache)")
    df = utils.load_exps([pasta / "acucar.parquet"])
    taxas, _ = utils.get_rate(df, t_run=DURACAO_MS / 1000, n_run=N_TENTATIVAS)
    return taxas["acucar"]


if __name__ == "__main__":
    nosso = rodar_numba()
    original = rodar_brian2().reindex(nosso.index, fill_value=0.0)

    ativos = original[original > 0].index.union(nosso[nosso > 0].index)
    tabela = pd.DataFrame({"brian2": original[ativos], "numba": nosso[ativos]})
    tabela = tabela.sort_values("brian2", ascending=False)
    print(f"\nneurônios ativos: brian2={int((original > 0).sum())}  numba={int((nosso > 0).sum())}")
    print(tabela.head(25).round(1).to_string())
    r = np.corrcoef(tabela.brian2, tabela.numba)[0, 1]
    print(f"\ncorrelação das taxas (neurônios ativos): r = {r:.3f}")
    print(f"MN9 (probóscide): brian2 = {original[MN9]:.1f} Hz   numba = {nosso[MN9]:.1f} Hz")
