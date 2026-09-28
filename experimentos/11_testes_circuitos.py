"""Testes de circuitos no cérebro corrigido (só cérebro, sem corpo).

1. Lobo antenal: com e sem as sinapses químicas excitatórias laterais,
   quão específico é o código de cheiro nas células de Kenyon.
2. Lado do cheiro: cheiro só na antena esquerda x só na direita -> DNa.
3. Reforço: açúcar -> dopamina de recompensa (PAM)? amargo -> punição (PPL1)?
4. Memória -> comportamento: MBONs de aproximar/evitar -> neurônios descendentes.
"""

import numpy as np

from mosca import biologia
from mosca.cerebro import Cerebro
from mosca.memoria import ODORES, CorpoCogumelo
from mosca.neuronios import ACUCAR_SHIU, Catalogo


def taxas(c, estimulos, ms=500, rep=2):
    tx = np.zeros(c.n)
    for s in range(rep):
        c.taxa_estimulo[:] = 0
        for idx, hz in estimulos:
            c.estimular_idx(idx, hz)
        c.reiniciar(s)
        c.rodar(50)
        c.zerar_contagem()
        c.rodar(ms)
        tx += c.taxas() / rep
    return tx


def codigo_olfativo(c, cat, rotulo):
    kc = c.idx(cat.ids(cell_class="Kenyon_Cell"))
    ativos = {o: taxas(c, [(c.idx(cat.orn(g)), 250)])[kc] > 2 for o, g in ODORES.items()}
    sobre = lambda a, b: (ativos[a] & ativos[b]).sum() / max((ativos[a] | ativos[b]).sum(), 1)  # noqa: E731
    print(f"  {rotulo:32s} KCs ativos {np.mean([a.mean() for a in ativos.values()]):5.1%} | "
          f"sobreposição A-B {sobre('A', 'B'):.2f}  A-C {sobre('A', 'C'):.2f}  B-C {sobre('B', 'C'):.2f}")  # fmt: skip


print("1. Lobo antenal -> células de Kenyon")
c = Cerebro(seed=0)
cat = Catalogo(c.con.ids)
biologia.aplicar(c, cat)
codigo_olfativo(c, cat, "lobo real (sem laterais químicas)")
c_lat = Cerebro(seed=0)  # mesmas correções, mas mantendo as laterais químicas
biologia.aplicar(c_lat, cat, lobo_real=False)
codigo_olfativo(c_lat, cat, "lobo com as laterais químicas")
del c_lat

print("\n2. Lado do cheiro -> neurônios de virar (DNa01/DNa02)")
dna = {lado: c.idx(cat.tipo("DNa01", lado) + cat.tipo("DNa02", lado)) for lado in ("left", "right")}
for odor in ("C", "A"):
    for lado in ("left", "right"):
        tx = taxas(c, [(c.idx(cat.orn(ODORES[odor], lado)), 250)])
        print(f"  cheiro {odor} só na antena {lado:5s}: DNa esquerda {tx[dna['left']].mean():5.1f} Hz | direita {tx[dna['right']].mean():5.1f} Hz")

print("\n3. Reforço -> dopamina")
pam = c.idx(cat.ids(cell_type=lambda t: t.startswith("PAM")))
ppl1 = c.idx(cat.ids(cell_type=lambda t: t.startswith("PPL1")))
for nome, idx in (("açúcar", c.idx(ACUCAR_SHIU)), ("amargo", c.idx(cat.amargo()))):
    tx = taxas(c, [(idx, 200)])
    print(f"  {nome}: PAM (recompensa) {tx[pam].mean():.2f} Hz | PPL1 (punição) {tx[ppl1].mean():.2f} Hz")

print("\n4. MBONs -> neurônios descendentes")
mb = CorpoCogumelo(c, cat, repouso_mbon=-46.0)
dn = c.idx(cat.descendentes)
for nome, grupo in (("aproximar", mb.mbon[mb.sinal_mbon > 0]), ("evitar", mb.mbon[mb.sinal_mbon < 0])):
    tx = taxas(c, [(grupo, 40)])[dn]
    print(f"  MBONs de {nome}: {(tx > 2).sum()} descendentes ativos (> 2 Hz)")
