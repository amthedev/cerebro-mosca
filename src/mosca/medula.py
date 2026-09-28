"""Medula (cordão nervoso ventral) real: conectoma BANC (Bates et al. 2026).

O BANC é outro indivíduo (fêmea) mapeado com cérebro + medula juntos. Aqui
usamos só a parte da medula, com os neurônios descendentes (DNs) como entrada:
os DNs do nosso cérebro FlyWire são casados com os do BANC pelo tipo celular
(o BANC traz a correspondência com o FlyWire em `fafb_cell_type`).

Saída: os motoneurônios de perna, cada um anotado com a perna e a função
(ex.: "flex_femur_tibia_joint"), que correspondem às articulações do corpo.

Mesma dinâmica LIF do cérebro (mosca.cerebro), com os mesmos cuidados:
moduladores fora da transmissão rápida, sinal pelo neurotransmissor conhecido.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mosca.cerebro import PASTA_DADOS, Cerebro, Conectoma

PASTA_BANC = PASTA_DADOS / "banc"
RAPIDOS = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0}
PERNAS = {"front_leg": "f", "middle_leg": "m", "hind_leg": "h"}


def carregar_medula(min_sinapses: int = 3) -> tuple[Conectoma, pd.DataFrame]:
    cache = PASTA_BANC / f"medula_{min_sinapses}.npz"
    meta = pd.read_feather(PASTA_BANC / "banc_888_meta.feather")
    meta["banc_888_id"] = meta.banc_888_id.astype(str)
    na_medula = (meta.region == "ventral_nerve_cord") | (meta.super_class == "descending")
    neur = meta[na_medula & ~meta.super_class.isin(["glia", "not_a_neuron", "trachea"])].reset_index(drop=True)
    if cache.exists():
        d = np.load(cache)
        neur = neur.set_index("banc_888_id").loc[d["ids"].astype(str)].reset_index()
        return Conectoma(d["ids"], d["indptr"], d["alvos"], d["sinapses"]), neur

    pos = pd.Series(np.arange(len(neur)), index=neur.banc_888_id)
    arestas = pd.read_feather(PASTA_BANC / "banc_888_edgelist_simple_v2.feather", columns=["pre", "post", "count"])
    arestas = arestas[(arestas["count"] >= min_sinapses) & arestas.pre.isin(pos.index) & arestas.post.isin(pos.index)]
    pre = pos.loc[arestas.pre].to_numpy()
    post = pos.loc[arestas.post].to_numpy()
    cont = arestas["count"].to_numpy()

    # sinal por neurônio: verificado na literatura > previsto (confiança >= 0.5)
    nt = neur.neurotransmitter_verified.fillna("").astype(str).str.split("[,;]").str[0].str.strip()
    prev = neur.neurotransmitter_predicted.where(neur.neurotransmitter_score >= 0.5, "")
    nt = nt.where(nt != "", prev.fillna(""))
    sinal = nt.map(RAPIDOS).fillna(0.0).to_numpy().copy()  # moduladores/desconhecidos: 0
    sinal[(neur.super_class == "descending").to_numpy() & (sinal == 0)] = 1.0  # DNs: maioria colinérgica
    sinal[(neur.super_class == "motor").to_numpy()] = 0.0  # motoneurônios só saem para os músculos
    # exceto o PSI ("interneurônio de sinapse periférica"): colinérgico, faz
    # sinapse nos motoneurônios dos músculos de voo DLM (King & Wyman 1980)
    sinal[(neur.cell_type == "PSI").to_numpy()] = 1.0

    ordem = np.argsort(pre, kind="stable")
    n = len(neur)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(pre, minlength=n), out=indptr[1:])
    con = Conectoma(
        ids=neur.banc_888_id.astype(np.int64).to_numpy(),
        indptr=indptr,
        alvos=post[ordem].astype(np.int32),
        sinapses=(cont[ordem] * sinal[pre[ordem]]).astype(np.int32),
    )
    np.savez(cache, ids=con.ids, indptr=con.indptr, alvos=con.alvos, sinapses=con.sinapses)
    return con, neur


def adicionar_conexoes(con: Conectoma, pares: list[tuple[int, int, int]]) -> Conectoma:
    """Novo conectoma com conexões extras (pré, pós, sinapses equivalentes)."""
    n = con.n
    pre = np.repeat(np.arange(n), np.diff(con.indptr))
    pre = np.concatenate([pre, [p for p, _, _ in pares]])
    alvos = np.concatenate([con.alvos, np.array([q for _, q, _ in pares], dtype=np.int32)])
    sin = np.concatenate([con.sinapses, np.array([w for _, _, w in pares], dtype=np.int32)])
    ordem = np.argsort(pre, kind="stable")
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(pre, minlength=n), out=indptr[1:])
    return Conectoma(con.ids, indptr, alvos[ordem].astype(np.int32), sin[ordem].astype(np.int32))


# Sinapses elétricas bem estabelecidas que o conectoma (só químico) não mostra,
# e uma sinapse química identificada com eficácia 1:1 medida:
#   Fibra Gigante -> TTMn (músculo de salto) e -> PSI: junções comunicantes
#   (King & Wyman 1980; Allen et al. 2006, ShakB);
#   PSI -> motoneurônios dos músculos de voo DLM: colinérgica, segue 1:1
#   (King & Wyman 1980); o peso uniforme do modelo a subestima.
ELETRICAS = [("DNp01", "jump_escape"), ("DNp01", "PSI"), ("PSI", "DLM")]
PESO_ELETRICA = 400  # sinapses equivalentes: cada disparo da Fibra Gigante faz o TTMn disparar (1:1)


class Medula:
    def __init__(self, min_sinapses: int = 3, seed: int | None = 0, eletricas: bool = True):
        self.con, self.neur = carregar_medula(min_sinapses)
        n = self.neur
        if eletricas:
            pares = []
            for tipo_pre, funcao_pos in ELETRICAS:
                for lado in ("left", "right"):
                    pre = n.index[(n.fafb_cell_type.fillna(n.cell_type) == tipo_pre) & (n.side == lado)]
                    tipo = n.cell_type.astype(str)
                    pos = n.index[((n.cell_function == funcao_pos) | tipo.str.startswith(funcao_pos)) & (n.side == lado)]
                    pares += [(int(a), int(b), PESO_ELETRICA) for a in pre for b in pos]
            self.eletricas = pares
            self.con = adicionar_conexoes(self.con, pares)
        self.c = Cerebro(self.con, seed=seed)
        self.c.dep_tau_ms = 500.0
        self.dn_por_tipo: dict[tuple[str, str], np.ndarray] = {}
        dns = n[n.super_class == "descending"]
        for (tipo, lado), g in dns.groupby([dns.fafb_cell_type.fillna(dns.cell_type), dns.side]):
            self.dn_por_tipo[(str(tipo), str(lado))] = g.index.to_numpy()
        pernas = n[n.cell_class == "leg_motor_neuron"]
        self.mn = pernas.index.to_numpy()
        self.mn_perna = (pernas.side.str[0] + pernas.body_part_effector.map(PERNAS)).to_numpy()  # ex.: "lf"
        self.mn_funcao = pernas.cell_function_detailed.fillna("unknown_leg_movement").to_numpy()
        self.salto = n.index[n.cell_function == "jump_escape"].to_numpy()
        self.voo = n.index[n.cell_type.astype(str).str.startswith("DLM")].to_numpy()  # músculos de voo (potência)
        self.psi = n.index[n.cell_type == "PSI"].to_numpy()  # via de fuga: Fibra Gigante -> PSI -> DLM
        # A saída da Fibra Gigante cansa com o uso (habituação da fuga). Sem
        # isso, as duas Fibras Gigantes, muito conectadas entre si no BANC,
        # ficam se disparando para sempre no modelo LIF.
        self.c.dep_U[self.dns("DNp01")] = 0.8

    def dns(self, tipo: str, lado: str | None = None) -> np.ndarray:
        lados = [lado] if lado else ["left", "right", "center", "na"]
        return np.concatenate([self.dn_por_tipo.get((tipo, l), np.array([], int)) for l in lados])

    def resposta(self, estimulos: dict[str, float] | list, ms: float = 400.0, seed: int = 0) -> pd.DataFrame:
        """Estimula DNs (índices -> Hz) e devolve taxa dos motoneurônios de perna."""
        c = self.c
        c.taxa_estimulo[:] = 0
        for idx, hz in estimulos:
            c.estimular_idx(idx, hz)
        c.reiniciar(seed)
        c.rodar(50)
        c.zerar_contagem()
        c.rodar(ms)
        tx = c.taxas()
        return pd.DataFrame({"perna": self.mn_perna, "funcao": self.mn_funcao, "hz": tx[self.mn]})


class AcopladorDescendentes:
    """Liga o cérebro FlyWire à medula BANC: cada neurônio descendente do BANC
    dispara na taxa do grupo do mesmo tipo e lado no cérebro FlyWire."""

    def __init__(self, cerebro: Cerebro, cat, medula: Medula, tau_ms: float = 20.0):
        self.cerebro, self.med, self.tau = cerebro, medula, tau_ms
        ann = cat.ann
        dn_fw = cerebro.idx(cat.descendentes)
        tipos_fw = ann.loc[cerebro.con.ids[dn_fw], ["cell_type", "side"]].astype(str)
        chaves = sorted(set(map(tuple, tipos_fw.to_numpy())) & set(medula.dn_por_tipo))
        grupo = {k: g for g, k in enumerate(chaves)}
        g_fw = np.array([grupo.get(tuple(k), -1) for k in tipos_fw.to_numpy()])
        self.fw_idx, self.fw_grupo = dn_fw[g_fw >= 0], g_fw[g_fw >= 0]
        self.banc_idx = np.concatenate([medula.dn_por_tipo[k] for k in chaves])
        self.banc_grupo = np.concatenate([np.full(len(medula.dn_por_tipo[k]), grupo[k]) for k in chaves])
        self.n_grupos = len(chaves)
        self.tamanho = np.bincount(self.fw_grupo, minlength=self.n_grupos).clip(1)
        self.taxa = np.zeros(self.n_grupos)
        self._ant = cerebro.contagem[self.fw_idx].astype(np.int64)
        self._ant_ttm = 0
        self._ant_dlm = 0
        self.disparos_voo = 0
        print(f"medula acoplada: {self.n_grupos} tipos de descendentes casados FlyWire<->BANC")

    def passo(self, ms: float) -> int:
        """Roda a medula por `ms` e devolve quantos disparos do TTMn (salto) houve."""
        agora = self.cerebro.contagem[self.fw_idx].astype(np.int64)
        d = np.maximum(agora - self._ant, 0)
        self._ant = agora
        hz = np.bincount(self.fw_grupo, weights=d, minlength=self.n_grupos) / self.tamanho / (ms * 1e-3)
        a = np.exp(-ms / self.tau)
        self.taxa = a * self.taxa + (1 - a) * hz
        c = self.med.c
        taxa = np.where(self.taxa < 1.0, 0.0, self.taxa)  # abaixo de 1 Hz: silêncio
        c.taxa_estimulo[self.banc_idx] = taxa[self.banc_grupo]
        c.rodar(ms)
        ttm = int(c.contagem[self.med.salto].sum())
        disparos, self._ant_ttm = ttm - self._ant_ttm, ttm
        # decolagem de fuga: PSI dispara (só a Fibra Gigante o ativa) e aciona os DLM.
        # Os DLM também recebem outros descendentes (ex.: os ativados por cheiro),
        # que sozinhos não iniciam o voo aqui.
        psi = int(c.contagem[self.med.psi].sum())
        self.disparos_voo, self._ant_dlm = psi - self._ant_dlm, psi
        return disparos
