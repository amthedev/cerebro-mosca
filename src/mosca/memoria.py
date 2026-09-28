"""Corpo cogumelo: memória associativa ensinada pela dopamina.

Tudo que é fiação vem do conectoma:
  - neurônios de projeção (PN) -> células de Kenyon (KC): código esparso do cheiro
  - KC -> neurônios de saída (MBON): as sinapses que aprendem
  - dopamina (DAN) -> MBON: diz qual DAN ensina qual compartimento

Regra de aprendizado (Hige et al. 2015, Cohn et al. 2015): quando uma KC esteve
ativa (traço de elegibilidade) e a dopamina do compartimento chega, a sinapse
KC -> MBON enfraquece. PAM = recompensa, PPL1 = punição.

Valência (Aso et al. 2014b): MBONs de compartimentos ensinados por PAM levam a
evitar; os de compartimentos PPL1 levam a se aproximar. Recompensa enfraquece os
de "evitar" -> o cheiro fica atraente; punição enfraquece os de "aproximar".

O cheiro entra pelos neurônios receptores olfativos (ORNs) reais de cada
antena e passa pelo lobo antenal inteiro (ver biologia.lobo_antenal_real).
Opção lobo="idealizado": atalho antigo, cada cheiro ativa direto os PNs.

Atalho: o reforço (açúcar, perigo) ativa os DANs diretamente.
"""

from __future__ import annotations

import numba as nb
import numpy as np
import scipy.sparse as sp

from mosca.cerebro import Cerebro
from mosca.neuronios import Catalogo

# Cheiros de teste: conjuntos de glomérulos que não se sobrepõem
ODORES = {
    "A": ("DC1", "DL5", "VA3", "VM3", "DL1", "VC1"),
    "B": ("DM3", "DM6", "VA6", "VC3", "VA5", "DC2"),
    "C": ("DM1", "DM2", "DM4", "DP1m", "VA2", "VM2"),  # comida fermentada
    "D": ("DA1", "VA1d", "VA1v", "DL3"),  # feromônios
}


@nb.njit(cache=True)
def _deprimir(pesos, idx_con, kc_local, mbon_local, elig, dopamina, eta, dt_s, w0, rec):
    for n in range(idx_con.shape[0]):
        j = idx_con[n]
        dw = eta * elig[kc_local[n]] * dopamina[mbon_local[n]] * dt_s
        w = pesos[j] * (1.0 - min(dw, 1.0))
        pesos[j] = w + (w0[n] - w) * rec


class CorpoCogumelo:
    def __init__(
        self,
        cerebro: Cerebro,
        cat: Catalogo,
        *,
        taxa_pn: float = 120.0,
        repouso_mbon: float = -44.5,  # acima do limiar: MBONs com atividade espontânea
        eta: float = 4e-5,
        tau_elig_ms: float = 500.0,
        tau_recuperacao_s: float = 3600.0,
        lobo: str = "real",
        taxa_orn: float = 250.0,
    ):
        self.c = cerebro
        self.cat = cat
        ann = cat.ann
        self.taxa_pn = taxa_pn
        self.eta = eta
        self.tau_elig_ms = tau_elig_ms
        self.tau_rec_s = tau_recuperacao_s

        self.kc = cerebro.idx(cat.ids(cell_class="Kenyon_Cell"))
        self.mbon = cerebro.idx(cat.ids(cell_class="MBON"))
        self.pam = cerebro.idx(cat.ids(cell_type=lambda t: t.startswith("PAM")))
        self.ppl1 = cerebro.idx(cat.ids(cell_type=lambda t: t.startswith("PPL1")))
        self.dan = np.concatenate([self.pam, self.ppl1])
        pn_ids = ann.index[ann.cell_class == "ALPN"]
        self.pn = cerebro.idx(pn_ids)
        self._tipo_pn = dict(zip(cerebro.idx(pn_ids), ann.loc[pn_ids, "cell_type"].astype(str)))

        self.lobo = lobo
        self.taxa_orn = taxa_orn
        self._orns = {  # ORNs de cada cheiro, por antena
            nome: [cerebro.idx(cat.orn(g, lado)) for lado in ("left", "right")] for nome, g in ODORES.items()
        }
        self._todos_orns = np.unique(np.concatenate([np.concatenate(v) for v in self._orns.values()]))
        if lobo == "idealizado":  # PNs viram relés (nenhuma entrada sináptica)
            eh_pn = np.zeros(cerebro.n, dtype=bool)
            eh_pn[self.pn] = True
            cerebro.pesos[eh_pn[cerebro.con.alvos]] = 0.0

        # MBONs têm atividade espontânea in vivo (~5-20 Hz); o cheiro modula
        cerebro.v_rep[self.mbon] = repouso_mbon

        # conexões KC -> MBON (as que aprendem)
        pos_kc = {int(k): n for n, k in enumerate(self.kc)}
        pos_mbon = {int(m): n for n, m in enumerate(self.mbon)}
        idx, kl, ml = [], [], []
        for k in self.kc:
            a, b = cerebro.con.indptr[k], cerebro.con.indptr[k + 1]
            for j in range(a, b):
                m = int(cerebro.con.alvos[j])
                if m in pos_mbon:
                    idx.append(j)
                    kl.append(pos_kc[int(k)])
                    ml.append(pos_mbon[m])
        self.idx_con = np.array(idx, dtype=np.int64)
        self.kc_local = np.array(kl, dtype=np.int64)
        self.mbon_local = np.array(ml, dtype=np.int64)
        self.w0 = cerebro.pesos[self.idx_con].copy()

        # DAN -> MBON pelo número de sinapses (independe dos pesos rápidos)
        linhas, cols, vals = [], [], []
        for d_local, d in enumerate(self.dan):
            a, b = cerebro.con.indptr[d], cerebro.con.indptr[d + 1]
            for j in range(a, b):
                m = int(cerebro.con.alvos[j])
                if m in pos_mbon:
                    linhas.append(pos_mbon[m])
                    cols.append(d_local)
                    vals.append(abs(int(cerebro.con.sinapses[j])))
        W = sp.csr_matrix((vals, (linhas, cols)), shape=(len(self.mbon), len(self.dan)), dtype=float)
        W.data[W.data < 5] = 0.0  # só conexões fortes (dentro do compartimento)
        W.eliminate_zeros()
        total = np.asarray(W.sum(axis=1)).ravel()
        self.W_dan = sp.diags(1.0 / np.maximum(total, 1)) @ W
        # valência de cada MBON: +1 aproximar (compartimento PPL1), -1 evitar
        # (compartimento PAM). Só MBONs típicos com um tipo de DAN dominante.
        de_pam = np.asarray(W[:, : len(self.pam)].sum(axis=1)).ravel()
        de_ppl1 = np.asarray(W[:, len(self.pam) :].sum(axis=1)).ravel()
        tipos = ann.loc[cerebro.con.ids[self.mbon], "cell_type"].astype(str).str.extract(r"MBON(\d+)")[0]
        tipico = tipos.fillna("99").astype(int).to_numpy() <= 25
        dominante = np.maximum(de_pam, de_ppl1) >= 0.75 * np.maximum(total, 1)
        self.sinal_mbon = np.where(
            tipico & dominante & (total >= 20), np.where(de_ppl1 > de_pam, 1.0, -1.0), 0.0
        )

        self.elig = np.zeros(len(self.kc))
        self.elig_rapida = np.zeros(len(self.kc))  # KCs ativas agora (~200 ms)
        self._ant_kc = np.zeros(len(self.kc), dtype=np.int64)
        self._ant_dan = np.zeros(len(self.dan), dtype=np.int64)
        self._ant_mbon = np.zeros(len(self.mbon), dtype=np.int64)
        self.taxa_mbon = np.zeros(len(self.mbon))
        self.base_mbon = np.zeros(len(self.mbon))  # média lenta (linha de base)
        self.dopamina = np.zeros(len(self.mbon))

    # ---- entradas -------------------------------------------------------
    def pns_do_odor(self, glomerulos) -> np.ndarray:
        return np.array(
            [i for i, t in self._tipo_pn.items() if any(t.startswith(g + "_") for g in glomerulos)],
            dtype=np.int64,
        )

    def cheirar(self, intensidades: dict) -> None:
        """Intensidade (0-1) de cada cheiro de ODORES: um número (as duas
        antenas iguais) ou um par (antena esquerda, direita)."""
        if self.lobo == "idealizado":
            taxa = np.zeros(self.c.n)
            for nome, x in intensidades.items():
                x = float(np.mean(x))
                if x > 0:
                    taxa[self.pns_do_odor(ODORES[nome])] = self.taxa_pn * min(x, 1.0)
            self.c.taxa_estimulo[self.pn] = taxa[self.pn]
            return
        self.c.taxa_estimulo[self._todos_orns] = 0.0
        for nome, x in intensidades.items():
            x = np.broadcast_to(np.asarray(x, dtype=float), (2,))
            for lado in range(2):
                if x[lado] > 0.02:
                    self.c.estimular_idx(self._orns[nome][lado], self.taxa_orn * min(x[lado], 1.0))

    def reforcar(self, recompensa: float = 0.0, punicao: float = 0.0) -> None:
        """Ativa dopamina de recompensa (PAM) e/ou punição (PPL1), 0-1."""
        self.c.estimular_idx(self.pam, 60.0 * recompensa)
        self.c.estimular_idx(self.ppl1, 60.0 * punicao)

    # ---- aprendizado e leitura -------------------------------------------
    def atualizar(self, janela_ms: float, aprender: bool = True) -> None:
        s = janela_ms * 1e-3
        cont = self.c.contagem
        kc_now = cont[self.kc].astype(np.int64)
        dan_now = cont[self.dan].astype(np.int64)
        mb_now = cont[self.mbon].astype(np.int64)
        # contagem pode ter sido zerada por fora
        d_kc = np.maximum(kc_now - self._ant_kc, 0)
        d_dan = np.maximum(dan_now - self._ant_dan, 0)
        d_mb = np.maximum(mb_now - self._ant_mbon, 0)
        self._ant_kc, self._ant_dan, self._ant_mbon = kc_now, dan_now, mb_now

        self.elig = self.elig * np.exp(-janela_ms / self.tau_elig_ms) + d_kc
        self.elig_rapida = self.elig_rapida * np.exp(-janela_ms / 200.0) + d_kc
        self.dopamina = self.W_dan @ (d_dan / s)
        a = np.exp(-janela_ms / 100.0)
        self.taxa_mbon = a * self.taxa_mbon + (1 - a) * d_mb / s
        a = np.exp(-janela_ms / 3000.0)
        self.base_mbon = a * self.base_mbon + (1 - a) * self.taxa_mbon
        if aprender:
            _deprimir(
                self.c.pesos, self.idx_con, self.kc_local, self.mbon_local,
                self.elig, self.dopamina, self.eta, s, self.w0, s / self.tau_rec_s,
            )  # fmt: skip

    def zerar_leitura(self) -> None:
        self.elig[:] = 0.0
        self.elig_rapida[:] = 0.0
        self._ant_kc[:] = self.c.contagem[self.kc]
        self._ant_dan[:] = self.c.contagem[self.dan]
        self._ant_mbon[:] = self.c.contagem[self.mbon]

    def valencia(self, taxas: np.ndarray | None = None, base: np.ndarray | None = None) -> float:
        """Quanto o cheiro atual muda os MBONs de "aproximar" contra os de
        "evitar", em relação à linha de base. > 0 atrai, < 0 repele (-1 a 1)."""
        r = self.taxa_mbon if taxas is None else taxas
        b = self.base_mbon if base is None else base
        d = r - b
        return float((self.sinal_mbon * d).sum() / (np.abs(self.sinal_mbon * d).sum() + 20.0))

    def valencia_aprendida(self, janela_ms: float = 300.0) -> float:
        """O que o corpo cogumelo lembra do cheiro atual: mudança na entrada dos
        MBONs de aproximar menos a dos de evitar, para as KCs ativas agora
        (traço recente). > 0: associado a recompensa; < 0: a punição."""
        ativo = self.elig_rapida
        if ativo.sum() <= 0:
            return 0.0
        r = ativo[self.kc_local]
        w = np.abs(self.c.pesos[self.idx_con])
        w0 = np.abs(self.w0)
        delta = np.bincount(self.mbon_local, weights=r * (w - w0), minlength=len(self.mbon))
        total = np.bincount(self.mbon_local, weights=r * w0, minlength=len(self.mbon))
        return float((self.sinal_mbon * delta).sum() / max(total.sum(), 1e-9) * 10.0)

    def forca_memoria(self) -> float:
        """Fração média das sinapses KC->MBON enfraquecidas."""
        return float(1 - np.abs(self.c.pesos[self.idx_con]).sum() / np.abs(self.w0).sum())
