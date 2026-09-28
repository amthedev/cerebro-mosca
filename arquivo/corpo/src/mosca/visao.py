"""Visão pela retina: olhos compostos do NeuroMechFly (721 omatídeos por olho).

O MuJoCo renderiza o que cada olho vê (lente olho-de-peixe) e o FlyGym agrega
os pixels em omatídeos. Daqui sai o sinal de aproximação (looming) que vai
para os neurônios LPLC2 de cada lado.

Atalho: o detector de expansão (fração do céu que ficou escura e a velocidade
com que ela cresce) está no lugar dos circuitos T4/T5 -> LPLC2 da lobula.
O que é real: a resolução do olho (~5 graus por omatídeo, então um predador
pequeno e longe é invisível), oclusão, iluminação e o campo de visão de cada olho.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


class Olhos:
    def __init__(self, corpo, fracao_ceu: float = 0.4, luz: float = 1.0):
        self.corpo = corpo
        sim = corpo.sim
        sim.get_ommatidia_readouts(corpo.fly.name)  # cria a retina e o renderizador dos olhos
        ret = sim.retina
        ids = ret.ommatidia_id_map
        n = int(ids.max())
        linhas = ndimage.mean(np.indices(ids.shape)[0], labels=ids, index=np.arange(1, n + 1))
        self.ceu = linhas < fracao_ceu * ret.nrows  # omatídeos que olham acima do horizonte
        self.luz = luz  # 1 = dia, ~0 = noite
        self.sinal = np.zeros(2)
        self.escuro = np.zeros(2)
        self.ultima_imagem = None
        self._t_ant = None
        self._proxima = 0.0

    def ver(self) -> np.ndarray:
        """Luminância de cada omatídeo (2 olhos x 721), já com a luz do ambiente."""
        r = self.corpo.sim.get_ommatidia_readouts(self.corpo.fly.name).max(axis=2)
        self.ultima_imagem = r * self.luz
        return self.ultima_imagem

    def atualizar(self, t: float) -> np.ndarray:
        """Sinal de aproximação (0-1) em cada olho. Amostra a retina a cada
        30 ms, ou a cada 10 ms quando há algo escuro no céu."""
        if t < self._proxima:
            return self.sinal
        # compensação do próprio movimento: com o corpo inclinado o horizonte
        # gira e o chão invade o céu; nesse caso só atualiza a referência
        c = self.corpo
        z = c.sim.mj_data.xmat[c._torax_id].reshape(3, 3)[:, 2]
        # cópia eferente: durante o próprio salto e logo depois, a mudança na
        # imagem é causada pela própria mosca e não conta como ameaça
        if c.saltando or c.voando:
            self._silencio_ate = t + 0.5
        inclinado = z[2] < np.cos(np.radians(20)) or t < getattr(self, "_silencio_ate", -1.0)
        r = self.ver()
        ceu = r[:, self.ceu]
        base = np.median(ceu, axis=1, keepdims=True)
        enxerga = base.max() > 0.05  # no escuro não há contraste suficiente
        escuro = (ceu < 0.5 * base).mean(axis=1) if enxerga else np.zeros(2)
        if enxerga != getattr(self, "_enxergava", enxerga) or inclinado or getattr(self, "_inclinado", False):
            self.escuro = escuro  # luz mudou ou corpo inclinado: recomeça a referência
        self._enxergava = enxerga
        self._inclinado = inclinado
        dt = 0.03 if self._t_ant is None else max(t - self._t_ant, 1e-3)
        crescimento = (escuro - self.escuro) / dt  # fração do céu por segundo
        # como os LPLC2, exige expansão em duas leituras seguidas
        anterior = getattr(self, "_cresc_ant", np.zeros(2))
        self._cresc_ant = crescimento
        expansao = np.clip(np.minimum(crescimento, anterior) / 1.5, 0, 1)
        perto = np.clip((escuro - 0.02) / 0.2, 0, 1) * ((crescimento > 0) & (anterior > 0))
        novo = np.maximum(expansao, 0.5 * perto)
        if inclinado:
            novo = np.zeros(2)
        self.sinal = np.maximum(novo, self.sinal * np.exp(-dt / 0.05))
        self.escuro, self._t_ant = escuro, t
        self._proxima = t + (0.01 if escuro.max() > 0.001 else 0.03)
        return self.sinal

    def imagem_legivel(self, olho: int = 0) -> np.ndarray:
        """Imagem do mosaico de omatídeos (para mostrar no vídeo)."""
        if self.ultima_imagem is None:
            return None
        ret = self.corpo.sim.retina
        return ret.hex_pxls_to_human_readable(np.clip(self.ultima_imagem[olho], 0, 1), color_8bit=True)
