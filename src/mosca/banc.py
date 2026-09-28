"""Cérebro da segunda mosca: a parte cerebral do conectoma BANC.

O BANC (Bates, Phelps, Kim, Yang et al. 2026) é outra fêmea, mapeada com
cérebro + medula juntos. Aqui pegamos o equivalente ao volume do FlyWire
(cérebro e lobos ópticos, mais os neurônios que sobem da medula, que no
FlyWire aparecem cortados no pescoço), com a mesma dinâmica LIF e as mesmas
regras de sinal, para rodar os mesmos experimentos nas duas moscas e comparar
tipo celular por tipo celular.

Cada neurônio do BANC recebe o nome do seu tipo no FlyWire (`tipo`):
  1. pelo neurônio casado no FlyWire (`fafb_match`), quando existe;
  2. senão pelo `fafb_cell_type`, traduzido para o nome das anotações FlyWire
     que usamos quando a tradução é consistente (quando os dois existem, 97%
     coincidem; o BANC às vezes divide um tipo, ex.: LB3 -> LB3a-d).

O sinal de cada neurônio segue a regra do modelo original (Shiu et al.):
GABA e glutamato previstos inibem, o resto excita. As correções biológicas
entram depois, com o mesmo `biologia.aplicar` do FlyWire, via `CatalogoBANC`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mosca.cerebro import PASTA_DADOS, Conectoma
from mosca.neuronios import ARQUIVO_ANOTACOES, Catalogo

PASTA_BANC = PASTA_DADOS / "banc"
NAO_NEURONIOS = {"glia", "not_a_neuron", "trachea"}
DA_MEDULA_AO_CEREBRO = {"ascending", "sensory_ascending", "ascending_visceral_circulatory"}
INIBITORIOS = {"gaba", "glutamate"}
# classes do BANC com o nome usado nas anotações FlyWire (as que biologia.py usa)
CLASSES = {
    "kenyon_cell": "Kenyon_Cell",
    "mushroom_body_dopaminergic_neuron": "DAN",
    "antennal_lobe_projection_neuron": "ALPN",
    "antennal_lobe_local_neuron": "ALLN",
    "olfactory_receptor_neuron": "olfactory",
}


def _moda_consistente(s: pd.Series, minimo: float = 0.8):
    cont = s.value_counts()
    return cont.index[0] if cont.iloc[0] >= minimo * cont.sum() else np.nan


def metadados() -> pd.DataFrame:
    """Neurônios do BANC equivalentes ao volume do FlyWire, com o tipo FlyWire."""
    meta = pd.read_feather(PASTA_BANC / "banc_888_meta.feather")
    meta = meta[~meta.super_class.isin(NAO_NEURONIOS)]
    no_cerebro = (meta.region != "ventral_nerve_cord") | meta.super_class.isin(DA_MEDULA_AO_CEREBRO)
    meta = meta[no_cerebro].reset_index(drop=True)
    meta["id"] = meta.banc_888_id.astype(str).astype(np.int64)
    # ~13 mil neurônios (quase todos do lobo óptico) vêm sem lado: pela posição,
    # longe da linha média (esquerda ~185 mil nm, direita ~58 mil nm, meio ~121 mil)
    x = pd.to_numeric(meta.position.astype(str).str.split(",").str[0], errors="coerce")
    sem_lado = meta.side.isna()
    meta.loc[sem_lado & (x > 150_000), "side"] = "left"
    meta.loc[sem_lado & (x < 75_000), "side"] = "right"

    fw = pd.read_csv(ARQUIVO_ANOTACOES, sep="\t", usecols=["root_id", "cell_type"], dtype={"root_id": str})
    fw = fw.set_index("root_id").cell_type
    casado = meta.fafb_match.astype("string").str.split(",").str[0].str.strip()  # ids como texto: sem perder dígitos
    pelo_casado = casado.map(fw)
    nome = meta.fafb_cell_type.astype("string").str.replace("auto:", "", regex=False)
    ambos = pd.DataFrame({"nome": nome, "fw": pelo_casado}).dropna()
    traducao = ambos.groupby("nome").fw.agg(_moda_consistente).dropna()
    meta["tipo"] = pelo_casado.fillna(nome.map(traducao)).fillna(nome).astype(object)
    meta["tipo_por"] = np.where(pelo_casado.notna(), "neurônio casado", np.where(nome.notna(), "nome do tipo", ""))
    return meta


def carregar_cerebro(meta: pd.DataFrame | None = None, sinapses: str = "v2") -> Conectoma:
    """Conectoma do cérebro BANC, com o sinal pela regra do modelo original.

    sinapses: "v2" (tamanho >= 5 voxels; a tabela do artigo do BANC) ou "v3"
    (detector novo, tamanho >= 10; recomendada pelo BANC para trabalhos novos).
    """
    cache = PASTA_BANC / f"cerebro_banc_{sinapses}.npz"
    if cache.exists():
        d = np.load(cache)
        return Conectoma(d["ids"], d["indptr"], d["alvos"], d["sinapses"])
    meta = metadados() if meta is None else meta
    pos = pd.Series(np.arange(len(meta)), index=meta.banc_888_id.astype(str))
    e = pd.read_feather(PASTA_BANC / f"banc_888_edgelist_simple_{sinapses}.feather", columns=["pre", "post", "count"])
    e = e[e.pre.isin(pos.index) & e.post.isin(pos.index) & (e.pre != e.post)]  # o FlyWire do modelo não tem autapses
    pre = pos.loc[e.pre].to_numpy()
    post = pos.loc[e.post].to_numpy()
    sinal = np.where(meta.neurotransmitter_predicted.isin(INIBITORIOS), -1, 1)
    ordem = np.argsort(pre, kind="stable")
    n = len(meta)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(pre, minlength=n), out=indptr[1:])
    con = Conectoma(
        ids=meta.id.to_numpy(),
        indptr=indptr,
        alvos=post[ordem].astype(np.int32),
        sinapses=(e["count"].to_numpy()[ordem] * sinal[pre[ordem]]).astype(np.int32),
    )
    np.savez(cache, ids=con.ids, indptr=con.indptr, alvos=con.alvos, sinapses=con.sinapses)
    return con


def razao_sinapses(con_a: Conectoma, cat_a: Catalogo, con_b: Conectoma, cat_b: Catalogo) -> float:
    """Mediana, entre os tipos presentes nas duas moscas, da razão das sinapses
    de entrada por neurônio (a / b). Mede a diferença de detecção de sinapses
    entre os dois conectomas; serve para igualar o peso por sinapse."""

    def por_tipo(con, cat):
        entrada = np.bincount(con.alvos, weights=np.abs(con.sinapses), minlength=con.n)
        tipo = cat.ann.cell_type.reindex(con.ids).to_numpy()
        return pd.Series(entrada).groupby(tipo).mean()

    a, b = por_tipo(con_a, cat_a), por_tipo(con_b, cat_b)
    comum = a.index.intersection(b.index)
    r = a[comum] / b[comum]
    return float(r[np.isfinite(r) & (r > 0)].median())


def fatores_por_tipo(
    con: Conectoma, cat: Catalogo, con_ref: Conectoma, cat_ref: Catalogo, limites=(0.5, 8.0)
) -> np.ndarray:
    """Fator por neurônio pós-sináptico que iguala as sinapses de entrada do
    seu tipo às do mesmo tipo no conectoma de referência.

    A falta de sinapses do BANC não é uniforme (ex.: ~3x nas entradas do MN9,
    ~2x no cérebro em geral), então um fator único leva partes do cérebro à
    explosão enquanto outras continuam fracas. Multiplicando as entradas de
    cada neurônio pelo fator do seu tipo, a proporção entre as entradas (e o
    equilíbrio excitação/inibição) fica a do próprio BANC; só a escala total
    vem da referência. Tipos que só existem no BANC recebem a mediana.
    """

    def entrada_por_tipo(c, k):
        entrada = np.bincount(c.alvos, weights=np.abs(c.sinapses), minlength=c.n)
        tipo = k.ann.cell_type.reindex(c.ids).to_numpy()
        return pd.Series(entrada).groupby(tipo).mean(), tipo

    ref, _ = entrada_por_tipo(con_ref, cat_ref)
    proprio, tipo = entrada_por_tipo(con, cat)
    razao = (ref / proprio).dropna()
    razao = razao[np.isfinite(razao) & (razao > 0)]
    fator = pd.Series(tipo).map(razao).fillna(float(razao.median())).to_numpy()
    return np.clip(fator, *limites).astype(np.float32)


def sinal_por_tipo(pesos: np.ndarray, con: Conectoma, cat: Catalogo) -> pd.Series:
    """Sinal mais comum de cada tipo celular (+1 excita, -1 inibe, 0 modulador)."""
    pre = np.repeat(np.arange(con.n), np.diff(con.indptr))
    soma = np.zeros(con.n)
    np.add.at(soma, pre, np.sign(pesos))
    tem_saida = np.diff(con.indptr) > 0
    tipo = cat.ann.cell_type.reindex(con.ids).to_numpy()
    s = pd.Series(np.sign(soma)[tem_saida]).groupby(tipo[tem_saida])
    return s.agg(lambda x: x.mode().iloc[0])


def usar_sinais(cerebro, cat: Catalogo, sinais: pd.Series) -> int:
    """Dá a cada neurônio o sinal do seu tipo em `sinais` (de outra mosca),
    com o peso cheio do conectoma. Tipos ausentes mantêm o próprio sinal.
    Aplicar depois das correções e refazer `biologia.lobo_antenal_real`.
    Devolve quantos neurônios mudaram."""
    con = cerebro.con
    tipo = cat.ann.cell_type.reindex(con.ids).to_numpy()
    novo = pd.Series(tipo).map(sinais).to_numpy(dtype=float)
    mudou = 0
    for i in np.flatnonzero(~np.isnan(novo)):
        a, b = con.indptr[i], con.indptr[i + 1]
        if b > a:
            w = cerebro.pesos[a:b]
            mudou += int(np.any(np.sign(w) != novo[i]))
            w[:] = np.abs(con.sinapses[a:b]) * np.float32(cerebro.p.w_syn * novo[i])
    return mudou


class CatalogoBANC(Catalogo):
    """Mesma interface do catálogo FlyWire, sobre os neurônios do BANC.

    `cell_type` é o tipo FlyWire (para comparar as moscas); `banc_type` é o
    nome do próprio BANC (mais fino em alguns casos).

    consenso_tipo=True: a previsão de neurotransmissor de cada neurônio vira a
    da maioria do seu tipo BANC (confiança = fração que concorda), como a
    regra por tipo, mas só com dados do BANC. Não usa a coluna
    `cell_type_neurotransmitter_predicted` do arquivo de previsões: ela
    contradiz a maioria dos neurônios em tipos grandes (ver 17_nt_banc.py).
    """

    def __init__(self, ids_modelo: np.ndarray | None = None, meta: pd.DataFrame | None = None, consenso_tipo: bool = False):
        meta = metadados() if meta is None else meta
        if ids_modelo is not None:
            meta = meta[meta.id.isin(ids_modelo)]
        if consenso_tipo:
            meta = meta.copy()
            tem = meta.cell_type.notna() & meta.neurotransmitter_predicted.notna()
            cont = meta[tem].groupby("cell_type").neurotransmitter_predicted.value_counts(normalize=True)
            maioria = cont.groupby(level=0).idxmax().map(lambda x: x[1])
            fracao = cont.groupby(level=0).max()
            meta.loc[tem, "neurotransmitter_predicted"] = meta.loc[tem, "cell_type"].map(maioria)
            meta.loc[tem, "neurotransmitter_score"] = meta.loc[tem, "cell_type"].map(fracao)
        sensorial = meta.super_class.astype(str).str.startswith("sensory")
        classe = meta.cell_class.map(CLASSES)
        classe = classe.where(classe.notna() | ~sensorial, "unknown_sensory").fillna(meta.cell_class)
        self.meta = meta
        self.ann = pd.DataFrame(
            {
                "cell_type": meta.tipo.to_numpy(),
                "banc_type": meta.cell_type.to_numpy(),
                "side": meta.side.to_numpy(),
                "super_class": meta.super_class.to_numpy(),
                "cell_class": classe.to_numpy(),
                "cell_sub_class": meta.cell_sub_class.to_numpy(),
                "known_nt": meta.neurotransmitter_verified.to_numpy(),
                "top_nt": meta.neurotransmitter_predicted.to_numpy(),
                "top_nt_conf": meta.neurotransmitter_score.to_numpy(),
            },
            index=pd.Index(meta.id.to_numpy(), name="root_id"),
        )

    def mn9(self, lado=None):
        return self.ids(lado, banc_type="MN9")
