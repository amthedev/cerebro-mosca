"""Mesmo modelo, duas moscas: o cérebro prevê a mesma coisa em indivíduos diferentes?

FlyWire (fêmea 1) x BANC (fêmea 2, só a parte do cérebro). Mesmo estímulo (o
mesmo tipo celular FlyWire nas duas moscas) e mesma leitura (taxa média de
cada tipo celular, por lado). Três comparações por protocolo:
  ruído        mesma mosca, mesmo lado, outra semente (o teto possível)
  hemisférios  mesma mosca, estímulo à esquerda x à direita espelhado
               (as duas metades do cérebro já são dois circuitos diferentes)
  indivíduos   FlyWire x BANC, mesmo lado
Medidas, sem contar os tipos estimulados: correlação de Spearman das taxas
entre os tipos que respondem (> 2 Hz) em pelo menos uma das duas condições, e
Jaccard (fração dos que respondem nas duas).
Os dois cérebros com as correções de biologia.py. O BANC em sete versões:
  como está        pesos da tabela de sinapses v2 do BANC (tamanho >= 5,
                   a do artigo do BANC)
  escala global    peso por sinapse x a razão mediana de sinapses (~2,1x:
                   a tabela v2 tem ~metade das sinapses nos mesmos tipos)
  calibrado        entradas de cada neurônio x o fator do seu tipo
                   (a razão varia por região)
  calibrado + consenso BANC   idem, com o neurotransmissor de cada neurônio
                   vindo da maioria do seu tipo no próprio BANC
  calibrado + sinais do FlyWire   idem, com o sinal de cada tipo vindo do
                   FlyWire (as previsões discordam em ~7% dos tipos)
  v3 ...           as mesmas, com a tabela v3 (detector novo, tamanho >= 10)
Saídas-chave: MN9 (açúcar), Fibra Gigante (LPLC2), aBN1 = SAD093 (antena).

Saída: resultados/duas_moscas.parquet (taxa por tipo em cada condição).
"""

import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mosca import banc, biologia
from mosca.cerebro import RAIZ, Cerebro, Parametros, carregar_conectoma
from mosca.neuronios import Catalogo

MS = 1000  # como uma tentativa do artigo original
LIMIAR_HZ = 2.0
AMARGO = ("LB1a,LB1d", "LB1b", "LB1c")  # LB1e: o FlyWire chama de amargo, o BANC de Ir94e (aminoácidos)
PROTOCOLOS = {
    "açúcar/água (LB3)": (lambda cat, lado: cat.ids(lado, cell_type="LB3"), 150),
    "amargo (LB1a-d)": (lambda cat, lado: cat.ids(lado, cell_type=AMARGO), 150),
    "cheiro de comida": (lambda cat, lado: cat.olfato_comida(lado), 250),
    "CO2 (ORN_V)": (lambda cat, lado: cat.orn(("V",), lado), 250),
    "ameaça (LPLC2)": (lambda cat, lado: cat.lplc2(lado), 150),
    "antena (JO-C/E)": (lambda cat, lado: cat.ids(lado, cell_type=lambda t: t.startswith(("JO-C", "JO-E"))), 150),
}
ESPELHO = {"left": "right", "right": "left"}


SAIDAS = {"açúcar/água (LB3)": "MN9", "ameaça (LPLC2)": "DNp01", "antena (JO-C/E)": "SAD093"}


class Mosca:
    def __init__(self, nome, con, cat, c):
        self.nome, self.cat, self.c = nome, cat, c
        ann = cat.ann.reindex(con.ids)
        chave = pd.DataFrame({"tipo": ann.cell_type.to_numpy(), "lado": ann.side.fillna("?").to_numpy()})
        gb = chave.groupby(["tipo", "lado"])
        self.grupos = gb.ngroup().fillna(-1).astype(int).to_numpy()  # -1: sem tipo
        self.chaves = gb.size().index
        self.saidas = {p: c.idx(cat.mn9() if t == "MN9" else cat.ids(cell_type=t)) for p, t in SAIDAS.items()}

    def rodar(self, protocolo, lado, seed):
        escolher, hz = PROTOCOLOS[protocolo]
        c = self.c
        c.taxa_estimulo[:] = 0
        c.estimular_idx(c.idx(escolher(self.cat, lado)), hz)
        c.reiniciar(seed)
        c.zerar_contagem()
        c.rodar(MS)
        tx = c.taxas()
        ok = self.grupos >= 0
        soma = np.bincount(self.grupos[ok], weights=tx[ok], minlength=len(self.chaves))
        n = np.bincount(self.grupos[ok], minlength=len(self.chaves))
        s = pd.Series(soma / n, index=self.chaves)
        s.attrs["ativos"] = int((tx > LIMIAR_HZ).sum())
        s.attrs["saida"] = float(tx[self.saidas[protocolo]].mean()) if protocolo in self.saidas else np.nan
        return s


def espelhar(s):
    tipos, lados = s.index.get_level_values(0), s.index.get_level_values(1)
    return pd.Series(s.to_numpy(), index=pd.MultiIndex.from_arrays([tipos, lados.map(lambda x: ESPELHO.get(x, x))]))


def comparar(a, b, excluir):
    comum = a.index.intersection(b.index)
    comum = comum[~comum.get_level_values(0).isin(excluir)]
    x, y = a[comum].to_numpy(), b[comum].to_numpy()
    resp = (x > LIMIAR_HZ) | (y > LIMIAR_HZ)
    if resp.sum() < 5:
        return np.nan, np.nan, int(resp.sum())
    rho = spearmanr(x[resp], y[resp]).statistic
    jac = ((x > LIMIAR_HZ) & (y > LIMIAR_HZ)).sum() / resp.sum()
    return rho, jac, int(resp.sum())


t0 = time.time()
fw_con = carregar_conectoma()
fw_cat = Catalogo(fw_con.ids)
b_meta = banc.metadados()
b_cons = {v: banc.carregar_cerebro(b_meta, v) for v in ("v2", "v3")}
b_con = b_cons["v2"]
b_cat = banc.CatalogoBANC(b_con.ids, b_meta)
b_cat_consenso = banc.CatalogoBANC(b_con.ids, b_meta, consenso_tipo=True)
razao = banc.razao_sinapses(fw_con, fw_cat, b_con, b_cat)
print(f"razão v3: {banc.razao_sinapses(fw_con, fw_cat, b_cons['v3'], b_cat):.2f}")
print(f"FlyWire: {fw_con.n} neurônios, {len(fw_con.alvos) / 1e6:.1f} mi conexões | "
      f"BANC (cérebro): {b_con.n} neurônios, {len(b_con.alvos) / 1e6:.1f} mi conexões")  # fmt: skip
print(f"sinapses de entrada por neurônio, mesmos tipos: FlyWire/BANC = {razao:.2f} (mediana)")
for nome, (escolher, _) in PROTOCOLOS.items():
    print(f"  {nome:20s} estimulados à esquerda: FlyWire {len(escolher(fw_cat, 'left')):4d} | BANC {len(escolher(b_cat, 'left')):4d}")


VERSOES = {  # nome: (tabela de sinapses, escala, sinais)
    "como está": ("v2", 1.0, "próprios"),
    "escala global": ("v2", "global", "próprios"),
    "calibrado": ("v2", "tipo", "próprios"),
    "calibrado + consenso BANC": ("v2", "tipo", "consenso"),
    "calibrado + sinais do FlyWire": ("v2", "tipo", "flywire"),
    "v3 como está": ("v3", 1.0, "próprios"),
    "v3 calibrado + consenso BANC": ("v3", "tipo", "consenso"),
}


def cerebro_banc(versao):
    tabela, escala, sinais = VERSOES[versao]
    con = b_cons[tabela]
    c = Cerebro(con, seed=0, params=Parametros(w_syn=0.275 * (razao if escala == "global" else 1.0)))
    biologia.aplicar(c, b_cat_consenso if sinais == "consenso" else b_cat)
    if sinais == "flywire":
        banc.usar_sinais(c, b_cat, banc.sinal_por_tipo(fw.c.pesos, fw_con, fw_cat))
        biologia.lobo_antenal_real(c, b_cat)
    if escala == "tipo":
        c.pesos *= banc.fatores_por_tipo(con, b_cat, fw_con, fw_cat)[con.alvos]
    return c, con


c_fw = Cerebro(fw_con, seed=0)
biologia.aplicar(c_fw, fw_cat)
fw = Mosca("FlyWire", fw_con, fw_cat, c_fw)
CONDICOES = [("FlyWire", "left", 0), ("FlyWire", "left", 1), ("FlyWire", "right", 0)]
r_fw = {(p, lado, seed): fw.rodar(p, lado, seed) for p in PROTOCOLOS for _, lado, seed in CONDICOES}
print(f"FlyWire pronto ({time.time() - t0:.0f} s)")

linhas, resumo = [], []
for p in PROTOCOLOS:
    for (_, lado, seed) in CONDICOES:
        s = r_fw[p, lado, seed]
        linhas.append(pd.DataFrame({"versao": "FlyWire", "mosca": "FlyWire", "protocolo": p, "lado_estimulo": lado, "seed": seed,
                                    "tipo": s.index.get_level_values(0), "lado": s.index.get_level_values(1), "hz": s.to_numpy()}))  # fmt: skip
for versao in VERSOES:
    c_b, con_b = cerebro_banc(versao)
    bm = Mosca("BANC", con_b, b_cat, c_b)
    print(f"\n=== BANC {versao} ({time.time() - t0:.0f} s)")
    for protocolo, (escolher, _) in PROTOCOLOS.items():
        excluir = set(fw_cat.ann.loc[escolher(fw_cat, None), "cell_type"]) | set(b_cat.ann.loc[escolher(b_cat, None), "cell_type"])
        r = {("FlyWire", lado, seed): r_fw[protocolo, lado, seed] for _, lado, seed in CONDICOES}
        for _, lado, seed in CONDICOES:
            s = bm.rodar(protocolo, lado, seed)
            r["BANC", lado, seed] = s
            linhas.append(pd.DataFrame({"versao": versao, "mosca": "BANC", "protocolo": protocolo, "lado_estimulo": lado, "seed": seed,
                                        "tipo": s.index.get_level_values(0), "lado": s.index.get_level_values(1), "hz": s.to_numpy()}))  # fmt: skip
        comp = {
            "ruído FlyWire": comparar(r["FlyWire", "left", 0], r["FlyWire", "left", 1], excluir),
            "ruído BANC": comparar(r["BANC", "left", 0], r["BANC", "left", 1], excluir),
            "hemisférios FlyWire": comparar(r["FlyWire", "left", 0], espelhar(r["FlyWire", "right", 0]), excluir),
            "hemisférios BANC": comparar(r["BANC", "left", 0], espelhar(r["BANC", "right", 0]), excluir),
            "indivíduos (esq)": comparar(r["FlyWire", "left", 0], r["BANC", "left", 0], excluir),
            "indivíduos (dir)": comparar(r["FlyWire", "right", 0], r["BANC", "right", 0], excluir),
        }
        f, bb = r["FlyWire", "left", 0], r["BANC", "left", 0]
        saida = f" | {SAIDAS[protocolo]}: FlyWire {f.attrs['saida']:.1f} Hz, BANC {bb.attrs['saida']:.1f} Hz" if protocolo in SAIDAS else ""
        print(f"  {protocolo}  (neurônios ativos: FlyWire {f.attrs['ativos']}, BANC {bb.attrs['ativos']}){saida}")
        for nome, (rho, jac, n) in comp.items():
            print(f"    {nome:20s} Spearman {rho:5.2f} | Jaccard {jac:4.2f} | n={n}")
            resumo.append({"versao": versao, "protocolo": protocolo, "comparação": nome, "spearman": rho, "jaccard": jac, "n": n,
                           "ativos_fw": f.attrs["ativos"], "ativos_banc": bb.attrs["ativos"],
                           "saida_fw": f.attrs["saida"], "saida_banc": bb.attrs["saida"]})  # fmt: skip

pd.concat(linhas).to_parquet(RAIZ / "resultados" / "duas_moscas.parquet")
res = pd.DataFrame(resumo)
res.to_csv(RAIZ / "resultados" / "duas_moscas_resumo.csv", index=False)
print("\nMédia entre protocolos (Spearman / Jaccard):")
print(res.assign(comp=res["comparação"].str.replace(r" \(.*\)", "", regex=True))
      .groupby(["versao", "comp"], sort=False)[["spearman", "jaccard"]].mean().round(2).to_string())  # fmt: skip
print(f"\ntotal {time.time() - t0:.0f} s")
