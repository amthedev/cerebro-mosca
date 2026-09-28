"""Correções biológicas ao modelo de Shiu et al.

1. Moduladores: o modelo original trata dopamina, serotonina e octopamina como
   excitação rápida. Eles agem por receptores metabotrópicos (modulação lenta).
   Como excitação rápida, o neurônio serotonérgico CSD, por exemplo, liga o lobo
   antenal inteiro com qualquer cheiro. Aqui eles saem da transmissão rápida (a
   dopamina é usada no aprendizado, em memoria.py).
2. Sinais: alguns neurônios GABA/glutamatérgicos estão como excitatórios no
   arquivo original. O sinal passa a seguir o neurotransmissor conhecido da
   literatura (known_nt) ou, na falta dele, a previsão por imagem (top_nt).
   Cuidado: a previsão por imagem erra classes inteiras (diz que as 5.172
   células de Kenyon são dopaminérgicas; elas são colinérgicas). Por isso
   moduladores só são aceitos por identidade (classe DAN ou known_nt), e a
   previsão só é usada com confiança alta.
3. Lobo antenal: sem as sinapses químicas excitatórias laterais (neurônio
   local colinérgico -> PN e PN -> PN). Na mosca a excitação lateral entre
   glomérulos vem sobretudo de junções elétricas fracas (Yaksi & Wilson 2010;
   Huang et al. 2010); como sinapse química cheia no modelo LIF ela inunda o
   lobo antenal inteiro. Sem elas, cada cheiro ativa só os PNs dos seus
   glomérulos (80-93% contra 1% dos outros) e as células de Kenyon ficam
   esparsas (1,5-4%) e específicas, com toda a inibição local mantida.
4. Histamina inibe: o modelo original a trata como excitação, mas ela abre
   canais de cloreto (ort, HisCl1) e hiperpolariza o alvo (Hardie 1989;
   Gengs et al. 2002). No FlyWire são os fotorreceptores (R1-8) e poucos
   outros, todos por neurotransmissor conhecido.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mosca.cerebro import Cerebro
from mosca.neuronios import Catalogo

RAPIDOS = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0}
MODULADORES = {"dopamine", "serotonin", "octopamine"}
CONHECIDOS = set(RAPIDOS) | MODULADORES | {"histamine"}
SENSORIAIS_E_KC = {
    "Kenyon_Cell", "olfactory", "visual", "mechanosensory", "gustatory",
    "unknown_sensory", "hygrosensory", "thermosensory",
}  # fmt: skip


def nt_principal(texto) -> str | None:
    """Primeiro neurotransmissor clássico listado em known_nt (ignora '-negative')."""
    if not isinstance(texto, str):
        return None
    for parte in texto.replace(";", ",").split(","):
        p = parte.strip()
        if p in CONHECIDOS:
            return p
    return None


def classificar(cat: Catalogo, conf_rapido: float = 0.5, conf_modulador: float = 0.8) -> pd.Series:
    """Neurotransmissor de cada neurônio anotado (ou ausente, se incerto)."""
    ann = cat.ann
    conhecido = ann.known_nt.map(nt_principal)
    previsto = ann.top_nt.where(ann.top_nt_conf >= conf_rapido)
    previsto = previsto.where(previsto.isin(set(RAPIDOS) | {"histamine"}))
    mod_previsto = ann.top_nt.where(
        (ann.top_nt_conf >= conf_modulador)
        & ann.top_nt.isin(MODULADORES)
        & ~ann.cell_class.isin(SENSORIAIS_E_KC)
    )
    nt = conhecido.fillna(previsto).fillna(mod_previsto)
    nt[ann.cell_class == "DAN"] = "dopamine"
    nt[ann.cell_class == "Kenyon_Cell"] = "acetylcholine"  # Barnstedt et al. 2016
    return nt


def lobo_antenal_real(cerebro: Cerebro, cat: Catalogo) -> int:
    """Remove as sinapses químicas excitatórias laterais do lobo antenal."""
    pn = cerebro.idx(cat.ids(cell_class="ALPN"))
    locais = cerebro.idx(cat.ids(cell_class="ALLN"))
    eh_pn = np.zeros(cerebro.n, dtype=bool)
    eh_pn[pn] = True
    removidas = 0
    for i in np.concatenate([locais, pn]):
        a, b = cerebro.con.indptr[i], cerebro.con.indptr[i + 1]
        m = eh_pn[cerebro.con.alvos[a:b]] & (cerebro.pesos[a:b] > 0)
        cerebro.pesos[a:b][m] = 0.0
        removidas += int(m.sum())
    return removidas


def aplicar(
    cerebro: Cerebro,
    cat: Catalogo,
    depressao_orn: bool = False,
    lobo_real: bool = True,
    moduladores: bool = True,
    sinais: bool = True,
    histamina: bool = True,
) -> dict:
    """Aplica as correções (cada uma pode ser desligada, para auditar o efeito)."""
    rapidos = RAPIDOS | ({"histamine": -1.0} if histamina else {})
    nt = classificar(cat)
    mudou, silenciados = 0, []
    for fid, n in nt.dropna().items():
        i = cerebro.indice.get(int(fid))
        if i is None:
            continue
        a, b = cerebro.con.indptr[i], cerebro.con.indptr[i + 1]
        if b == a:
            continue
        w = cerebro.pesos[a:b]
        if n in MODULADORES:
            if moduladores:
                silenciados.append(i)
                w[:] = 0.0
        elif n in rapidos and (sinais or n == "histamine"):
            novo = np.abs(w) * rapidos[n]
            if np.sign(w[0]) != np.sign(novo[0]):
                mudou += 1
            w[:] = novo
    if depressao_orn:
        cerebro.dep_U[cerebro.idx(cat.ids(cell_class="olfactory"))] = 0.5
        cerebro.dep_tau_ms = 300.0
    laterais = lobo_antenal_real(cerebro, cat) if lobo_real else 0
    return {"moduladores": np.array(silenciados), "sinais_corrigidos": mudou, "laterais_removidas": laterais}
