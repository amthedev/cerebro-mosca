"""Auditoria do modelo original (Shiu et al. 2024): o que cada correção muda?

Cada correção de biologia.py é ligada sozinha e depois todas juntas, e o
modelo é testado contra:
  A. previsões do artigo confirmadas em moscas reais
     - açúcar ativa MN9 (probóscide) e Rattle, Usnea, Clavicle e Fudog;
     - amargo forte elimina a resposta do MN9 ao açúcar forte;
     - Ir94e forte não elimina;
     - JO-CE ativa forte o aBN1 (limpar a antena); JO-F ativa fraco.
  B. fatos conhecidos da biologia que o artigo não testou
     - cheiro ativa poucas células de Kenyon (~5-10%; Honegger et al. 2011);
     - cheiro ativa os PNs dos seus glomérulos, não os dos outros;
     - objeto se aproximando (LPLC2) ativa a Fibra Gigante.
  C. simetria: o artigo estimulou o açúcar da esquerda; o da direita deveria
     dar uma resposta do mesmo tamanho.
Neurônios nomeados: ids do repositório original (sez_neurons.pickle, v630),
todos presentes sem mudança na v783.
"""

import pickle

import numpy as np
import pandas as pd

from mosca import biologia
from mosca.cerebro import PASTA_SHIU, RAIZ, Cerebro
from mosca.memoria import ODORES
from mosca.neuronios import ACUCAR_SHIU, Catalogo

HZ, MS, ATIVO = 150, 1000, 2.0
with open(PASTA_SHIU / "sez_neurons.pickle", "rb") as f:
    NOMES = pickle.load(f)
ABN1 = [720575940630907434]  # figures.ipynb do repositório original
IL3LN6_ESQ = 720575940632403986  # interneurônio GABAérgico do lobo antenal, excitatório no arquivo original
CONDICOES = {
    "original": None,
    "só moduladores": {"moduladores": True, "sinais": False, "histamina": False, "lobo_real": False},
    "só sinais": {"moduladores": False, "sinais": True, "histamina": False, "lobo_real": False},
    "só histamina": {"moduladores": False, "sinais": False, "histamina": True, "lobo_real": False},
    "só lobo antenal": {"moduladores": False, "sinais": False, "histamina": False, "lobo_real": True},
    "todas": {},
}

base = Cerebro(seed=0)
cat = Catalogo(base.con.ids)
pesos_originais = base.pesos.copy()
acucar = base.idx(ACUCAR_SHIU)
amargo = base.idx(cat.ids(cell_type=("LB1a,LB1d", "LB1b", "LB1c")))
ir94e = base.idx(cat.ids(cell_type="LB1e"))  # o FlyWire chama de amargo; o BANC e a literatura, de Ir94e
jo_ce = base.idx(cat.ids(cell_type=lambda t: t.startswith(("JO-C", "JO-E"))))
jo_f = base.idx(cat.ids(cell_type=lambda t: t.startswith("JO-F")))
mn9 = base.idx(cat.mn9())
nomeados = {n: base.idx(NOMES[n]) for n in ("rattle", "usnea", "clavicle", "fudog")}
abn1 = base.idx(ABN1)
kc = base.idx(cat.ids(cell_class="Kenyon_Cell"))
gf = base.idx(cat.tipo("DNp01"))
pn = cat.ann.loc[cat.ids(cell_class="ALPN")]


def rodar(estimulos, seed=0):
    base.taxa_estimulo[:] = 0
    for idx, hz in estimulos:
        base.estimular_idx(idx, hz)
    base.reiniciar(seed)
    base.zerar_contagem()
    base.rodar(MS)
    return base.taxas().copy()


def pns_do_cheiro(tx, glomerulos):
    """% dos PNs dos glomérulos do cheiro ativos e % dos PNs dos outros."""
    glom = pn.cell_type.astype(str).str.extract(r"^([A-Za-z0-9]+?)_")[0]  # ex.: DM1_lPN -> DM1
    certo = base.idx(pn.index[glom.isin(glomerulos)])
    outro = base.idx(pn.index[~glom.isin(glomerulos) & glom.notna()])
    return (tx[certo] > ATIVO).mean(), (tx[outro] > ATIVO).mean()


linhas, respostas = [], {}
for nome, opcoes in CONDICOES.items():
    base.pesos[:] = pesos_originais
    if opcoes is not None:
        biologia.aplicar(base, cat, **opcoes)
    so_acucar = rodar([(acucar, HZ)])
    lb3 = {lado: rodar([(base.idx(cat.ids(lado, cell_type="LB3")), HZ)]) for lado in ("left", "right")}
    n_lb3 = {lado: int((tx > ATIVO).sum()) for lado, tx in lb3.items()}
    com_amargo = rodar([(acucar, HZ), (amargo, HZ)])
    com_ir94e = rodar([(acucar, HZ), (ir94e, HZ)])
    tx_jo_ce, tx_jo_f = rodar([(jo_ce, HZ)]), rodar([(jo_f, HZ)])
    cheiro = rodar([(base.idx(cat.orn(ODORES["A"])), 250)])
    ameaca = rodar([(base.idx(cat.lplc2("left")), HZ)])
    respostas[nome] = {"açúcar": so_acucar, "JO-CE": tx_jo_ce, "cheiro": cheiro, "ameaça": ameaca}
    m = so_acucar[mn9].mean()
    certo, outro = pns_do_cheiro(cheiro, ODORES["A"])
    testes = {
        "MN9 com açúcar (Hz)": (m, m > ATIVO),
        **{f"{n} com açúcar (Hz)": (so_acucar[i].mean(), so_acucar[i].mean() > ATIVO) for n, i in nomeados.items()},
        "MN9 açúcar+amargo / só açúcar": (com_amargo[mn9].mean() / max(m, 1e-9), com_amargo[mn9].mean() < 0.1 * m),
        "MN9 açúcar+Ir94e / só açúcar": (com_ir94e[mn9].mean() / max(m, 1e-9), com_ir94e[mn9].mean() > 0.1 * m),
        "aBN1 com JO-CE (Hz)": (tx_jo_ce[abn1].mean(), tx_jo_ce[abn1].mean() > ATIVO),
        "aBN1 com JO-F (Hz)": (tx_jo_f[abn1].mean(), tx_jo_f[abn1].mean() < tx_jo_ce[abn1].mean()),
        "células de Kenyon ativas (%)": (100 * (cheiro[kc] > ATIVO).mean(), (cheiro[kc] > ATIVO).mean() < 0.15),
        "PNs certos / outros ativos (%)": (f"{100 * certo:.0f} / {100 * outro:.0f}", certo > 0.5 and outro < 0.1),
        "Fibra Gigante com LPLC2 (Hz)": (ameaca[gf].mean(), ameaca[gf].mean() > ATIVO),
        "LB3 esq / dir: neurônios ativos": (f"{n_lb3['left']} / {n_lb3['right']}", n_lb3["right"] < 3 * n_lb3["left"]),
    }
    for teste, (valor, ok) in testes.items():
        linhas.append({"condição": nome, "teste": teste, "valor": valor, "ok": bool(ok)})
    print(f"{nome:16s} passou {sum(ok for _, ok in testes.values())}/{len(testes)}")

res = pd.DataFrame(linhas)
res.to_csv(RAIZ / "resultados" / "auditoria.csv", index=False)


def fmt(v):
    return f"{v:.2f}" if isinstance(v, float) else str(v)


tab = res.assign(v=[f"{fmt(v)} {'✓' if ok else '✗'}" for v, ok in zip(res.valor, res.ok)])
print("\n" + tab.pivot(index="teste", columns="condição", values="v")[list(CONDICOES)].to_string())

print("\nQuanto o cérebro inteiro muda em relação ao original (neurônios ativos; correlação das taxas):")
for estimulo in ("açúcar", "JO-CE", "cheiro", "ameaça"):
    o = respostas["original"][estimulo]
    partes = []
    for nome in list(CONDICOES)[1:]:
        t = respostas[nome][estimulo]
        resp = (o > ATIVO) | (t > ATIVO)
        r = np.corrcoef(o[resp], t[resp])[0, 1] if resp.sum() > 2 else np.nan
        partes.append(f"{nome}: {(t > ATIVO).sum()} (r={r:.2f})")
    print(f"  {estimulo:7s} original {(o > ATIVO).sum()} | " + " | ".join(partes))

print("\nAçúcar da direita no modelo original, corrigindo o sinal de um único neurônio:")
lb3_dir = base.idx(cat.ids("right", cell_type="LB3"))
base.pesos[:] = pesos_originais
antes = int((rodar([(lb3_dir, HZ)]) > ATIVO).sum())
i = base.indice[IL3LN6_ESQ]
a, b = base.con.indptr[i], base.con.indptr[i + 1]
base.pesos[a:b] = -np.abs(pesos_originais[a:b])
depois = int((rodar([(lb3_dir, HZ)]) > ATIVO).sum())
linha = cat.ann.loc[IL3LN6_ESQ]
print(f"  il3LN6 esquerdo ({IL3LN6_ESQ}; known_nt={linha.known_nt}, {b - a} conexões, "
      f"{int(np.abs(base.con.sinapses[a:b]).sum())} sinapses): {antes} -> {depois} neurônios ativos")
