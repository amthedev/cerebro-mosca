"""Valida o modo por eventos contra o modo denso e mede a velocidade.

Mesmo estímulo, mesma semente: os disparos de cada neurônio devem ser iguais
(o modo por eventos só pula neurônios em repouso exato; eps = limiar para
considerar que um neurônio voltou ao repouso).
"""

import time

import numpy as np

from mosca.cerebro import Cerebro, carregar_conectoma
from mosca.neuronios import ACUCAR_SHIU, Catalogo

con = carregar_conectoma()
cat = Catalogo(con.ids)
testes = {
    "açúcar (21 neurônios, 150 Hz)": lambda c: [(c.idx(ACUCAR_SHIU), 150)],
    "ameaça (LPLC2, 216 neurônios, 80 Hz)": lambda c: [(c.idx(cat.lplc2()), 80)],
    "cheiro (ORNs de 6 glomérulos, 100 Hz)": lambda c: [(c.idx(cat.olfato_comida()), 100)],
}
DUR = 1000.0
for nome, estim in testes.items():
    res = {}
    for modo, eps in (("denso", 0.0), ("eventos", 0.0), ("eventos", 1e-5), ("auto", 1e-5)):
        c = Cerebro(con, seed=0, modo=modo, eps=eps)
        for idx, hz in estim(c):
            c.estimular_idx(idx, hz)
        c.rodar(1.0)  # compila
        c.reiniciar(seed=7)
        t = time.time()
        for _ in range(int(DUR / 50)):  # em pedaços, como numa simulação com corpo
            c.rodar(50.0)
        dt = time.time() - t
        res[(modo, eps)] = (c.contagem.copy(), dt)
    ref, t_ref = res[("denso", 0.0)]
    print(f"\n{nome}: {int(ref.sum())} disparos no total")
    for (modo, eps), (cont, dt) in res.items():
        dif = np.abs(cont - ref)
        print(f"  {modo:7s} eps={eps:g}: {dt:6.2f} s por segundo simulado ({t_ref / dt:5.1f}x) | "
              f"neurônios com contagem diferente: {(dif > 0).sum()} | diferença total {int(dif.sum())} disparos")  # fmt: skip
