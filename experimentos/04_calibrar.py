"""Calibração: quanto açúcar dispara o MN9, com e sem cheiro de comida."""
import numpy as np
from mosca.cerebro import Cerebro
from mosca.neuronios import Catalogo, ACUCAR_SHIU
c=Cerebro(seed=0); cat=Catalogo(c.con.ids)
mn9=c.idx(cat.mn9()); ac=c.idx(ACUCAR_SHIU); orn=c.idx(cat.olfato_comida())
for cheiro in [0, 50]:
    res=[]
    for r in [60, 100, 150, 200, 300]:
        v=[]
        for seed in range(3):
            c.taxa_estimulo[:]=0; c.estimular_idx(ac, r); c.estimular_idx(orn, cheiro); c.reiniciar(seed); c.rodar(200); c.zerar_contagem(); c.rodar(1000); v.append(c.taxas()[mn9].mean())
        res.append((r, round(np.mean(v),1)))
    print('cheiro', cheiro, 'Hz | açúcar Hz -> MN9 Hz (1 s):', res)
