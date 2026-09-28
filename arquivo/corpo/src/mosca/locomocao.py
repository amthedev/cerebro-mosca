"""Controlador de caminhada rápido (medula virtual provisória).

Reimplementa a mesma matemática do HybridTurningController do FlyGym
(vendor/flygym_repo/src/flygym_demo/complex_terrain): seis osciladores
acoplados (CPG, marcha tripé), passos gravados de moscas reais e as regras de
correção de tropeço e de perna no ar. O original gasta ~2 ms de Python por
passo de 0,1 ms; aqui as tabelas são pré-calculadas e o laço roda em numba.

Recebe dois sinais descendentes (esquerda, direita): a força controla a
frequência dos passos e o tamanho do passo de cada lado; o sinal negativo faz
aquele lado andar para trás.
"""

from __future__ import annotations

import mujoco as mj
import numba as nb
import numpy as np

from flygym.anatomy import LEGS, BodySegment
from flygym_demo.complex_terrain.common import dof_spec_to_jointdof
from flygym_demo.complex_terrain.cpg_controller import get_cpg_biases
from flygym_demo.complex_terrain.hybrid_controller import (
    _CORRECTION_VECTORS,
    _DETECTED_STUMBLING_LINKS,
    _RIGHT_LEG_CORRECTION_SIGN,
)
from flygym_demo.complex_terrain.preprogrammed import PreprogrammedSteps

_N_TABELA = 4096


@nb.njit(cache=True)
def _passo(
    dt, sinal, fase, mag, freq_base, acoplamento, vies, convergencia,
    tabela, neutra, periodo_balanco, vetores_correcao,
    z_torax, z_tarsos, forca_proj_min,
    retracao, tropeco, persistencia,
    limiar_tropeco, limiar_retracao, taxas_retracao, taxas_tropeco,
    correcao_max, extensao_balanco, passos_persistencia, limiar_persistencia,
    saida_angulos, saida_adesao,
):  # fmt: skip
    n = 6
    dois_pi = 2 * np.pi
    # --- sinal descendente -> tamanho do passo e frequência de cada lado.
    # Como nas moscas, andar mais rápido é sobretudo dar passos mais
    # frequentes (~8 a ~24 passos/s); o tamanho do passo satura.
    amp = np.empty(n)
    freq = freq_base.copy()
    for p in range(n):
        s = sinal[0] if p < 3 else sinal[1]
        forca = abs(s)
        freq[p] = freq[p] * min(max(0.55 + 0.75 * forca, 0.4), 2.0)
        amp[p] = min(forca, 1.3)
        if s < 0:
            freq[p] = -freq[p]

    # --- regra de retração: perna muito mais alta que as outras
    altura = z_torax - z_tarsos
    ordem = np.argsort(altura)
    perna_retrair = -1
    if altura[ordem[-1]] > altura[ordem[-3]] + limiar_retracao:
        perna_retrair = ordem[-1]
        if retracao[perna_retrair] > limiar_persistencia:
            persistencia[perna_retrair] = 1
    for p in range(n):
        if persistencia[p] > 0:
            persistencia[p] += 1
        if persistencia[p] > passos_persistencia:
            persistencia[p] = 0

    # --- CPG (Euler)
    dfase = np.empty(n)
    for i in range(n):
        acc = 0.0
        for j in range(n):
            acc += mag[j] * acoplamento[i, j] * np.sin(fase[j] - fase[i] - vies[i, j])
        dfase[i] = dois_pi * freq[i] + acc
    for i in range(n):
        fase[i] += dfase[i] * dt
        mag[i] += convergencia[i] * (amp[i] - mag[i]) * dt

    # --- ângulos de cada perna
    for p in range(n):
        if p == perna_retrair or persistencia[p] > 0:
            retracao[p] += taxas_retracao[0] * dt
        else:
            retracao[p] = max(0.0, retracao[p] - taxas_retracao[1] * dt)
        if forca_proj_min[p] < limiar_tropeco:
            tropeco[p] += taxas_tropeco[0] * dt
        else:
            tropeco[p] = max(0.0, tropeco[p] - taxas_tropeco[1] * dt)
        if retracao[p] > 0:
            correcao = retracao[p]
            tropeco[p] = 0.0
        else:
            correcao = tropeco[p]
        correcao = min(max(correcao, 0.0), correcao_max)

        f = fase[p] % dois_pi
        x = f / dois_pi * (tabela.shape[2] - 1)
        k = min(int(x), tabela.shape[2] - 2)
        w = x - k
        ini, fim = periodo_balanco[p, 0], periodo_balanco[p, 1]
        # ganho da correção ao longo do ciclo (igual a _step_phase_gain)
        pts0 = ini
        pts1 = 0.5 * (ini + fim)
        pts2 = fim + extensao_balanco
        pts3 = 0.5 * (fim + dois_pi)
        if f <= pts0:
            ganho = 0.0
        elif f <= pts1:
            ganho = 0.8 * (f - pts0) / (pts1 - pts0)
        elif f <= pts2:
            ganho = 0.8 * (1 - (f - pts1) / (pts2 - pts1))
        elif f <= pts3:
            ganho = -0.1 * (f - pts2) / (pts3 - pts2)
        else:
            ganho = -0.1 * (1 - (f - pts3) / (dois_pi - pts3))
        for d in range(7):
            psi = tabela[p, d, k] * (1 - w) + tabela[p, d, k + 1] * w
            ang = neutra[p, d] + mag[p] * (psi - neutra[p, d])
            saida_angulos[p, d] = ang + correcao * ganho * vetores_correcao[p, d]
        saida_adesao[p] = not (ini < f < fim + extensao_balanco)


class ControladorPernas:
    def __init__(self, sim, fly, dofs_pernas, *, seed: int = 0):
        self.sim = sim
        self.dt = sim.timestep
        passos = PreprogrammedSteps()
        self.passos = passos

        fases = np.linspace(0, 2 * np.pi, _N_TABELA)
        self.tabela = np.stack([passos._psi_funcs[leg](fases) for leg in LEGS])
        self.neutra = np.stack([passos.neutral_pos[leg][:, 0] for leg in LEGS])
        self.periodo_balanco = np.stack([passos.swing_period[leg] for leg in LEGS])
        self.vetores = np.stack(
            [
                _CORRECTION_VECTORS[leg[1]] * (_RIGHT_LEG_CORRECTION_SIGN if leg[0] == "r" else 1)
                for leg in LEGS
            ]
        )
        # posição de cada (perna, dof) no vetor de atuadores das pernas
        pos = {d: i for i, d in enumerate(dofs_pernas)}
        self.idx_saida = np.array(
            [[pos[dof_spec_to_jointdof(leg, spec)] for spec in passos.dofs_per_leg] for leg in LEGS]
        )

        vies = get_cpg_biases("tripod")
        self.vies = vies
        self.acoplamento = (vies > 0) * 10.0
        self.freq_base = np.full(6, 12.0)
        self.convergencia = np.full(6, 20.0)

        # índices no MuJoCo para ler o corpo sem passar pelo Python do FlyGym
        segs = fly.get_bodysegs_order()
        ids = sim._internal_bodyids_by_fly[fly.name]
        self._id_torax = ids[segs.index(BodySegment("c_thorax"))]
        self._ids_tarsos = ids[[segs.index(BodySegment(f"{leg}_tarsus5")) for leg in LEGS]]
        geom_por_seg = sim._internal_geomid_by_bodyseg_by_fly[fly.name]
        self._linha_por_geom = np.full(sim.mj_model.ngeom, -1, dtype=np.int64)
        for p, leg in enumerate(LEGS):
            for link in _DETECTED_STUMBLING_LINKS:
                self._linha_por_geom[geom_por_seg[BodySegment(f"{leg}_{link}")]] = p
        self._eh_chao = np.zeros(sim.mj_model.ngeom, dtype=bool)
        self._eh_chao[sim._internal_ground_geom_ids] = True
        self._wrench = np.zeros(6)

        self.angulos = np.zeros((6, 7))
        self.adesao = np.zeros(6, dtype=np.bool_)
        self._saida = np.zeros(len(dofs_pernas))
        self.reset(seed)

    def reset(self, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.fase = rng.random(6) * 2 * np.pi
        self.mag = np.zeros(6)
        self.retracao = np.zeros(6)
        self.tropeco = np.zeros(6)
        self.persistencia = np.zeros(6, dtype=np.int64)

    def pose_neutra(self) -> np.ndarray:
        """Pose parada (fase pi, amplitude 1), na ordem dos atuadores."""
        f = np.full(6, np.pi)
        for p, leg in enumerate(LEGS):
            self._saida[self.idx_saida[p]] = self.passos.get_joint_angles(leg, f[p], 1.0)
        return self._saida.copy()

    def _forca_frontal_min(self, frente: np.ndarray) -> np.ndarray:
        """Menor força de contato com o chão na direção de avanço, por perna."""
        d = self.sim.mj_data
        min_proj = np.zeros(6)
        n = d.ncon
        if n == 0:
            return min_proj
        g1 = d.contact.geom1[:n]
        g2 = d.contact.geom2[:n]
        l1 = self._linha_por_geom[g1]
        l2 = self._linha_por_geom[g2]
        ativos = ((l1 >= 0) & self._eh_chao[g2]) | ((l2 >= 0) & self._eh_chao[g1])
        ativos &= ~d.contact.exclude[:n].astype(bool)
        # soma por segmento (como no original) e projeta na direção de avanço
        por_segmento = {}
        for c in np.flatnonzero(ativos):
            mj.mj_contactForce(self.sim.mj_model, d, int(c), self._wrench)
            f = d.contact.frame[c].reshape(3, 3).T @ self._wrench[:3]
            if l1[c] >= 0:
                chave, sinal = int(g1[c]), -1.0
            else:
                chave, sinal = int(g2[c]), 1.0
            por_segmento[chave] = por_segmento.get(chave, 0.0) + sinal * f
        for geom, f in por_segmento.items():
            p = self._linha_por_geom[geom]
            min_proj[p] = min(min_proj[p], float(f @ frente))
        return min_proj

    def step(self, sinal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = self.sim.mj_data
        frente = d.xmat[self._id_torax].reshape(3, 3)[:, 0].copy()
        _passo(
            self.dt, np.asarray(sinal, dtype=float), self.fase, self.mag,
            self.freq_base, self.acoplamento, self.vies, self.convergencia,
            self.tabela, self.neutra, self.periodo_balanco, self.vetores,
            float(d.xpos[self._id_torax, 2]), d.xpos[self._ids_tarsos, 2].copy(),
            self._forca_frontal_min(frente),
            self.retracao, self.tropeco, self.persistencia,
            -1.0, 0.05, np.array([800.0, 700.0]), np.array([2200.0, 1800.0]),
            80.0, np.pi / 4, 20, 20.0,
            self.angulos, self.adesao,
        )  # fmt: skip
        self._saida[self.idx_saida.ravel()] = self.angulos.ravel()
        return self._saida, self.adesao
