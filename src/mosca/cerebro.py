"""Cérebro da mosca: modelo LIF do conectoma FlyWire (Shiu et al., Nature 2024).

Mesmas equações, parâmetros e ordem de operações do modelo original em Brian2
(vendor/Drosophila_brain_model/model.py), reimplementado com numba para rodar
passo a passo em conjunto com o corpo físico.

Por passo de tempo (mesma agenda do Brian2):
  1. integra v e g (solução exata) nos neurônios fora do período refratário
  2. detecta disparos (v > v_th)
  3. entrega sinapses que chegam agora (atraso de 1.8 ms) e estímulos Poisson
  4. reseta quem disparou
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
PASTA_SHIU = RAIZ / "vendor" / "Drosophila_brain_model"
PASTA_DADOS = RAIZ / "dados"


@dataclass(frozen=True)
class Parametros:
    """Constantes do modelo (unidades: ms, mV, Hz). Valores de Shiu et al."""

    dt: float = 0.1
    v_0: float = -52.0  # potencial de repouso
    v_rst: float = -52.0  # potencial após disparo
    v_th: float = -45.0  # limiar de disparo
    t_mbr: float = 20.0  # constante de tempo da membrana
    tau: float = 5.0  # constante de tempo da sinapse
    t_rfc: float = 2.2  # período refratário
    t_dly: float = 1.8  # atraso sináptico
    w_syn: float = 0.275  # peso por sinapse (mV)
    f_poi: float = 250.0  # multiplicador do estímulo Poisson (garante disparo)


@dataclass
class Conectoma:
    """Grafo em formato CSR indexado pelo neurônio pré-sináptico."""

    ids: np.ndarray  # id FlyWire de cada neurônio (int64)
    indptr: np.ndarray  # int64, tamanho N+1
    alvos: np.ndarray  # int32, neurônio pós-sináptico de cada conexão
    sinapses: np.ndarray  # int32, nº de sinapses com sinal (+ excita, - inibe)

    @property
    def n(self) -> int:
        return len(self.ids)


def carregar_conectoma(versao: int = 783) -> Conectoma:
    """Carrega o conectoma, usando um cache .npz para acelerar as próximas vezes."""
    cache = PASTA_DADOS / f"conectoma_{versao}.npz"
    if cache.exists():
        d = np.load(cache)
        return Conectoma(d["ids"], d["indptr"], d["alvos"], d["sinapses"])

    comp = pd.read_csv(PASTA_SHIU / f"Completeness_{versao}.csv", index_col=0)
    con = pd.read_parquet(
        PASTA_SHIU / f"Connectivity_{versao}.parquet",
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    pre = con["Presynaptic_Index"].to_numpy()
    ordem = np.argsort(pre, kind="stable")
    n = len(comp)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(pre, minlength=n), out=indptr[1:])
    c = Conectoma(
        ids=comp.index.to_numpy(dtype=np.int64),
        indptr=indptr,
        alvos=con["Postsynaptic_Index"].to_numpy()[ordem].astype(np.int32),
        sinapses=con["Excitatory x Connectivity"].to_numpy()[ordem].astype(np.int32),
    )
    PASTA_DADOS.mkdir(exist_ok=True)
    np.savez(cache, ids=c.ids, indptr=c.indptr, alvos=c.alvos, sinapses=c.sinapses)
    return c


@nb.njit(cache=True)
def _semear(seed):
    np.random.seed(seed)


_BLOCO = 2048  # neurônios por bloco na parte paralela


@nb.njit(cache=True, parallel=True)
def _simular(
    n_passos, passo0,
    v, g, ultimo, fila, refr,
    indptr, alvos, pesos, silenciado,
    estim_idx, estim_prob, peso_poi, n_espont_medio,
    contagem, disparos_buf, n_disp_bloco,
    a, b, c, v_0, v_rst, v_th, atraso,
    dep_U, dep_x, dep_t, dep_tau,
):  # fmt: skip
    n = v.shape[0]
    tam_fila = fila.shape[0]
    n_blocos = n_disp_bloco.shape[0]
    for k in range(n_passos):
        t = passo0 + k
        # A chegada das sinapses do passo t-1 é aplicada aqui, no início do
        # passo t (equivale a aplicá-la no fim de t-1). Como no Brian2
        # ("unless refractory" = escrita condicional), o que chega a um
        # neurônio refratário em t-1 (inclusive quem disparou em t-1) se perde.
        anterior = fila[(t - 1) % tam_fila]

        # paralelo: chegada + integração + detecção de disparos
        for blk in nb.prange(n_blocos):
            ini = blk * _BLOCO
            fim = min(ini + _BLOCO, n)
            nd = 0
            for i in range(ini, fim):
                x = anterior[i]
                if x != 0.0:
                    anterior[i] = 0.0
                    if ultimo[i] != t - 1 and t - 1 - ultimo[i] >= refr[i]:
                        g[i] += x
                if t - ultimo[i] >= refr[i]:
                    u = v[i] - v_0[i]
                    v[i] = v_0[i] + u * a + g[i] * c
                    g[i] = g[i] * b
                    if v[i] > v_th[i]:
                        disparos_buf[ini + nd] = i
                        nd += 1
            n_disp_bloco[blk] = nd

        # serial: disparos entram na fila com atraso, estímulos e reset
        saida = fila[(t + atraso) % tam_fila]
        for blk in range(n_blocos):
            ini = blk * _BLOCO
            for s in range(ini, ini + n_disp_bloco[blk]):
                i = disparos_buf[s]
                if not silenciado[i]:
                    # depressão sináptica (Tsodyks-Markram, por neurônio pré):
                    # cada disparo gasta uma fração U do recurso, que se recupera
                    escala = 1.0
                    if dep_U[i] > 0.0:
                        xr = 1.0 - (1.0 - dep_x[i]) * np.exp(-(t - dep_t[i]) / dep_tau)
                        escala = xr
                        dep_x[i] = xr * (1.0 - dep_U[i])
                        dep_t[i] = t
                    for j in range(indptr[i], indptr[i + 1]):
                        saida[alvos[j]] += pesos[j] * escala
        for s in range(estim_idx.shape[0]):
            if np.random.random() < estim_prob[s]:
                v[estim_idx[s]] += peso_poi
        # atividade espontânea: disparos forçados em neurônios aleatórios
        if n_espont_medio > 0.0:
            for _ in range(np.random.poisson(n_espont_medio)):
                i = np.random.randint(0, n)
                if t - ultimo[i] >= refr[i] and ultimo[i] != t:
                    v[i] += peso_poi
        for blk in range(n_blocos):
            ini = blk * _BLOCO
            for s in range(ini, ini + n_disp_bloco[blk]):
                i = disparos_buf[s]
                v[i] = v_rst
                g[i] = 0.0
                ultimo[i] = t
                contagem[i] += 1
    return passo0 + n_passos


@nb.njit(cache=True)
def _simular_eventos(
    n_passos, passo0,
    v, g, ultimo, fila, marcado, tocados, n_tocados, refr,
    indptr, alvos, pesos, silenciado,
    estim_idx, estim_prob, peso_poi, n_espont_medio,
    contagem, ativo, lista, n_ativos, disparos_buf,
    a, b, c, v_rep, v_rst, v_th, atraso,
    dep_U, dep_x, dep_t, dep_tau, eps,
):  # fmt: skip
    """Mesma dinâmica de _simular, mas só calcula neurônios fora do repouso.

    Um neurônio em repouso exato (v = v_rep, g = 0, abaixo do limiar) não muda
    na integração, então pulá-lo é exato. A fila de sinapses guarda, para cada
    passo futuro, a lista de alvos tocados (não varre os 138 mil neurônios).
    Neurônios que voltam ao repouso (|v - v_rep| e |g| <= eps mV) saem da lista.
    """
    n = v.shape[0]
    tam_fila = fila.shape[0]
    na = n_ativos[0]
    for k in range(n_passos):
        t = passo0 + k
        # 1. chegada das sinapses do passo t-1 (mesma regra de _simular)
        sl = (t - 1) % tam_fila
        for q in range(n_tocados[sl]):
            i = tocados[sl, q]
            x = fila[sl, i]
            fila[sl, i] = 0.0
            marcado[sl, i] = 0
            if x != 0.0 and ultimo[i] != t - 1 and t - 1 - ultimo[i] >= refr[i]:
                g[i] += x
                if ativo[i] == 0:
                    ativo[i] = 1
                    lista[na] = i
                    na += 1
        n_tocados[sl] = 0

        # 2. integra os ativos, detecta disparos e tira quem voltou ao repouso
        nd = 0
        w = 0
        for q in range(na):
            i = lista[q]
            if t - ultimo[i] >= refr[i]:
                u = v[i] - v_rep[i]
                v[i] = v_rep[i] + u * a + g[i] * c
                g[i] = g[i] * b
                if v[i] > v_th[i]:
                    disparos_buf[nd] = i
                    nd += 1
            if (abs(v[i] - v_rep[i]) <= eps and abs(g[i]) <= eps
                    and v_rep[i] <= v_th[i] and v[i] <= v_th[i]):  # fmt: skip
                v[i] = v_rep[i]
                g[i] = 0.0
                ativo[i] = 0
            else:
                lista[w] = i
                w += 1
        na = w

        # 3. disparos entram na fila com atraso
        so = (t + atraso) % tam_fila
        for s in range(nd):
            i = disparos_buf[s]
            if not silenciado[i]:
                escala = 1.0
                if dep_U[i] > 0.0:
                    xr = 1.0 - (1.0 - dep_x[i]) * np.exp(-(t - dep_t[i]) / dep_tau)
                    escala = xr
                    dep_x[i] = xr * (1.0 - dep_U[i])
                    dep_t[i] = t
                for j in range(indptr[i], indptr[i + 1]):
                    alvo = alvos[j]
                    fila[so, alvo] += pesos[j] * escala
                    if marcado[so, alvo] == 0:
                        marcado[so, alvo] = 1
                        tocados[so, n_tocados[so]] = alvo
                        n_tocados[so] += 1

        # 4. estímulos e atividade espontânea (mesma ordem de sorteios)
        for s in range(estim_idx.shape[0]):
            if np.random.random() < estim_prob[s]:
                i = estim_idx[s]
                v[i] += peso_poi
                if ativo[i] == 0:
                    ativo[i] = 1
                    lista[na] = i
                    na += 1
        if n_espont_medio > 0.0:
            for _ in range(np.random.poisson(n_espont_medio)):
                i = np.random.randint(0, n)
                if t - ultimo[i] >= refr[i] and ultimo[i] != t:
                    v[i] += peso_poi
                    if ativo[i] == 0:
                        ativo[i] = 1
                        lista[na] = i
                        na += 1

        # 5. reset
        for s in range(nd):
            i = disparos_buf[s]
            v[i] = v_rst
            g[i] = 0.0
            ultimo[i] = t
            contagem[i] += 1
    n_ativos[0] = na
    return passo0 + n_passos


class Cerebro:
    """Cérebro inteiro da mosca, simulável em pedaços de tempo.

    Uso típico:
        cerebro = Cerebro()
        cerebro.estimular(ids_acucar, 150)   # Hz
        cerebro.rodar(1000)                  # ms
        taxas = cerebro.taxas()              # Hz desde o último zerar_contagem()
    """

    def __init__(
        self,
        conectoma: Conectoma | None = None,
        params: Parametros = Parametros(),
        seed: int | None = None,
        modo: str = "auto",
        eps: float = 1e-5,
    ):
        """modo "eventos": só calcula neurônios fora do repouso (~2x mais rápido
        com atividade esparsa); "denso": calcula todos a cada passo, em paralelo
        (melhor com atividade espalhada); "auto": escolhe a cada chamada de
        rodar() pela fração de neurônios fora do repouso. Os dois dão
        exatamente os mesmos disparos (experimentos/12_motor_eventos.py)."""
        self.con = conectoma or carregar_conectoma()
        self.p = params
        self.modo = modo
        self.eps = eps
        self.indice = {int(fid): i for i, fid in enumerate(self.con.ids)}

        p = params
        self._a = np.exp(-p.dt / p.t_mbr)
        self._b = np.exp(-p.dt / p.tau)
        self._c = p.tau / (p.tau - p.t_mbr) * (self._b - self._a)
        self._atraso = int(round(p.t_dly / p.dt))
        self._refr_padrao = int(round(p.t_rfc / p.dt))

        # pesos em mV; editáveis em tempo real (ex.: aprendizado)
        self.pesos = self.con.sinapses.astype(np.float32) * np.float32(p.w_syn)
        n = self.con.n
        self.silenciado = np.zeros(n, dtype=np.bool_)
        self.taxa_estimulo = np.zeros(n, dtype=np.float64)  # Hz por neurônio
        self.taxa_espontanea = 0.0  # Hz, disparos espontâneos em todo o cérebro
        self.v_th = np.full(n, p.v_th, dtype=np.float32)  # limiar por neurônio (mV)
        self.v_rep = np.full(n, p.v_0, dtype=np.float32)  # repouso por neurônio (mV)
        self.dep_U = np.zeros(n, dtype=np.float32)  # depressão sináptica (0 = sem)
        self.dep_tau_ms = 300.0
        self._disparos_buf = np.empty(n, dtype=np.int32)
        self._n_disp_bloco = np.zeros((n + _BLOCO - 1) // _BLOCO, dtype=np.int64)
        self._refr = np.full(n, self._refr_padrao, dtype=np.int32)
        self._estim_ant = np.zeros(0, dtype=np.int64)
        self.reiniciar(seed)

    @property
    def n(self) -> int:
        return self.con.n

    @property
    def tempo_ms(self) -> float:
        return self.passo * self.p.dt

    def reiniciar(self, seed: int | None = None) -> None:
        """Volta ao repouso (mantém pesos, estímulos e silenciamentos)."""
        n = self.n
        self.v = self.v_rep.copy()
        self.g = np.zeros(n, dtype=np.float32)
        self.ultimo = np.full(n, -(10**9), dtype=np.int64)
        self.fila = np.zeros((self._atraso + 1, n), dtype=np.float32)
        self.contagem = np.zeros(n, dtype=np.int32)
        self.dep_x = np.ones(n, dtype=np.float32)
        self.dep_t = np.zeros(n, dtype=np.int64)
        tam = self._atraso + 1  # estruturas do modo por eventos
        self._ativo = np.zeros(n, dtype=np.uint8)
        self._lista = np.zeros(n, dtype=np.int32)
        self._n_ativos = np.zeros(1, dtype=np.int64)
        self._marcado = np.zeros((tam, n), dtype=np.uint8)
        self._tocados = np.zeros((tam, n), dtype=np.int32)
        self._n_tocados = np.zeros(tam, dtype=np.int64)
        self.passo = 0
        self._passo_contagem = 0
        self._modo_atual = "denso"  # estado atual das estruturas (auto começa denso)
        _semear(np.random.SeedSequence(seed).generate_state(1)[0] % (2**31))

    # ---- endereçamento --------------------------------------------------
    def idx(self, ids_flywire) -> np.ndarray:
        """Converte ids FlyWire em índices internos (ignora ids ausentes)."""
        return np.array(
            [self.indice[int(f)] for f in ids_flywire if int(f) in self.indice],
            dtype=np.int64,
        )

    # ---- manipulações ---------------------------------------------------
    def estimular(self, ids_flywire, taxa_hz: float) -> None:
        """Estímulo tipo optogenética: disparos Poisson na taxa dada (0 desliga)."""
        self.taxa_estimulo[self.idx(ids_flywire)] = taxa_hz

    def estimular_idx(self, indices: np.ndarray, taxa_hz) -> None:
        self.taxa_estimulo[indices] = taxa_hz

    def silenciar(self, ids_flywire, ligado: bool = True) -> None:
        """Silencia neurônios: os disparos deles não chegam a ninguém."""
        self.silenciado[self.idx(ids_flywire)] = ligado

    # ---- simulação ------------------------------------------------------
    def _ativar_mudancas(self) -> None:
        """Modo por eventos: põe na lista quem está fora do repouso por mudança
        externa (v ou repouso alterados à mão) ou tem repouso acima do limiar."""
        fora = (self._ativo == 0) & ((self.v != self.v_rep) | (self.g != 0) | (self.v_rep > self.v_th))
        novos = np.flatnonzero(fora).astype(np.int32)
        if len(novos):
            na = int(self._n_ativos[0])
            self._lista[na : na + len(novos)] = novos
            self._ativo[novos] = 1
            self._n_ativos[0] = na + len(novos)

    def _trocar_modo(self, modo: str) -> None:
        """Sincroniza as estruturas ao trocar entre os modos."""
        if modo == self._modo_atual:
            return
        if modo == "denso":  # a fila densa já tem tudo; zera a contabilidade de eventos
            self._marcado[:] = 0
            self._n_tocados[:] = 0
            self._ativo[:] = 0
            self._n_ativos[0] = 0
        else:  # reconstrói as listas de alvos pendentes a partir da fila densa
            for sl in range(self.fila.shape[0]):
                nz = np.flatnonzero(self.fila[sl]).astype(np.int32)
                self._tocados[sl, : len(nz)] = nz
                self._n_tocados[sl] = len(nz)
                self._marcado[sl, nz] = 1
            self._ativo[:] = 0
            self._n_ativos[0] = 0
        self._modo_atual = modo

    def rodar(self, duracao_ms: float) -> None:
        n_passos = int(round(duracao_ms / self.p.dt))
        estim_idx = np.flatnonzero(self.taxa_estimulo > 0)
        estim_prob = self.taxa_estimulo[estim_idx] * self.p.dt * 1e-3
        # como no original: neurônios estimulados não têm período refratário
        if not np.array_equal(estim_idx, self._estim_ant):
            self._refr[self._estim_ant] = self._refr_padrao
            self._refr[estim_idx] = 0
            self._estim_ant = estim_idx
        refr = self._refr
        consts = (
            np.float32(self._a), np.float32(self._b), np.float32(self._c),
            self.v_rep, np.float32(self.p.v_rst), self.v_th, self._atraso,
            self.dep_U, self.dep_x, self.dep_t, self.dep_tau_ms / self.p.dt,
        )  # fmt: skip
        espont = self.taxa_espontanea * self.n * self.p.dt * 1e-3
        modo = self.modo
        if modo == "auto":
            fora = np.count_nonzero((self.v != self.v_rep) | (self.g != 0))
            modo = "eventos" if fora < 0.12 * self.n else "denso"
        self._trocar_modo(modo)
        if modo == "eventos":
            self._ativar_mudancas()
            self.passo = _simular_eventos(
                n_passos, self.passo,
                self.v, self.g, self.ultimo, self.fila, self._marcado, self._tocados, self._n_tocados, refr,
                self.con.indptr, self.con.alvos, self.pesos, self.silenciado,
                estim_idx, estim_prob, self.p.w_syn * self.p.f_poi, espont,
                self.contagem, self._ativo, self._lista, self._n_ativos, self._disparos_buf,
                *consts, self.eps,
            )  # fmt: skip
            return
        self.passo = _simular(
            n_passos, self.passo,
            self.v, self.g, self.ultimo, self.fila, refr,
            self.con.indptr, self.con.alvos, self.pesos, self.silenciado,
            estim_idx, estim_prob, self.p.w_syn * self.p.f_poi, espont,
            self.contagem, self._disparos_buf, self._n_disp_bloco,
            *consts,
        )  # fmt: skip

    def zerar_contagem(self) -> None:
        self.contagem[:] = 0
        self._passo_contagem = self.passo

    def taxas(self) -> np.ndarray:
        """Taxa de disparo (Hz) de cada neurônio desde o último zerar_contagem()."""
        janela_s = (self.passo - self._passo_contagem) * self.p.dt * 1e-3
        return self.contagem / max(janela_s, 1e-12)
