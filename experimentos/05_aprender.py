"""Condicionamento olfativo só no cérebro (como no labirinto T de laboratório).

Antes: mede a valência (atrai/repele) de 4 cheiros.
Treino: cheiro A + açúcar (dopamina PAM), cheiro B + punição (dopamina PPL1).
Depois: mede de novo. C e D nunca foram pareados (controle de especificidade).
"""

import numpy as np

from mosca import biologia
from mosca.cerebro import Cerebro
from mosca.memoria import ODORES, CorpoCogumelo
from mosca.neuronios import Catalogo

JANELA_MS = 10.0
cerebro = Cerebro(seed=0)
cat = Catalogo(cerebro.con.ids)
biologia.aplicar(cerebro, cat)
import sys

LOBO = sys.argv[1] if len(sys.argv) > 1 else "real"
mb = CorpoCogumelo(cerebro, cat, repouso_mbon=-46.0, eta=1e-4, lobo=LOBO)
print("lobo antenal:", LOBO)
print(f"KC->MBON que podem aprender: {len(mb.idx_con)} | MBONs de aproximar: {(mb.sinal_mbon > 0).sum()}, de evitar: {(mb.sinal_mbon < 0).sum()}, sem valência conhecida: {(mb.sinal_mbon == 0).sum()}")


def rodar(ms, odor=None, recompensa=0.0, punicao=0.0, aprender=True):
    mb.cheirar({odor: 1.0} if odor else {})
    mb.reforcar(recompensa, punicao)
    taxas = []
    for _ in range(int(ms / JANELA_MS)):
        cerebro.rodar(JANELA_MS)
        mb.atualizar(JANELA_MS, aprender=aprender)
        taxas.append(mb.taxa_mbon.copy())
    return np.mean(taxas[len(taxas) // 3 :], axis=0)


def testar(rotulo):
    res = {}
    for odor in ODORES:
        rodar(400, None, aprender=False)
        rodar(500, odor, aprender=False)
        res[odor] = (mb.valencia_aprendida(), 0.0)
    print(rotulo + "  ".join(f"{o}: {v:+.2f}" for o, (v, _) in res.items()))
    return res


antes = testar("ANTES   ")
for rep in range(5):
    rodar(1500, "A", recompensa=1.0)
    rodar(2500, None)
    rodar(1500, "B", punicao=1.0)
    rodar(2500, None)
    print(f"  treino {rep + 1}/5  memória: {mb.forca_memoria():.1%} das sinapses KC->MBON enfraquecidas")
depois = testar("DEPOIS  ")

print("\nMudança de valência (depois - antes):")
for o in ODORES:
    d = depois[o][0] - antes[o][0]
    papel = {"A": "pareado com açúcar", "B": "pareado com punição"}.get(o, "controle")
    print(f"  cheiro {o} ({papel:20s}): {d:+.2f}")
