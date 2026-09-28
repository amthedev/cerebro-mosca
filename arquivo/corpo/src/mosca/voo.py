"""Voo: asas batendo no ar, com aerodinâmica de asa de inseto.

Cada asa do NeuroMechFly tem 3 articulações. No frame da asa: "roll" faz a
varredura horizontal da batida, "pitch" gira a asa no próprio eixo (ângulo de
ataque) e "yaw" sobe/desce a asa.

Aerodinâmica: modelo de elemento de pá quase-estacionário com os coeficientes
medidos na asa-robô "Robofly" (Dickinson, Lehmann & Sane, Science 1999), que
já incluem a sustentação extra do vórtice no bordo de ataque:
    C_L(a) = 0.225 + 1.58 sin(2.13 a - 7.2°),  C_D(a) = 1.92 - 1.55 cos(2.04 a - 9.82°)
A cada passo, para cada asa: velocidade real do centro de pressão (a 70% da
envergadura), ângulo de ataque real, sustentação (perpendicular ao
movimento) e arrasto (contra o movimento), aplicados na própria asa.
(O modelo de fluido embutido do MuJoCo, quase-estacionário simples, dava só
~0,4x o peso: asas de inseto dependem desse efeito instável.)

Batida descrita como na biologia, no frame do tórax: ângulo de varredura phi
(no plano horizontal, 0 = asa para o lado) e ângulo de ataque alfa (rotação em
torno da envergadura), que vira de sinal na inversão da batida. Cinemática
inversa numérica (tabela) converte (phi, alfa) nos ângulos das 3 articulações
da asa, que são encadeadas e mudam de eixo conforme a asa abre.

Controle de voo (atalho: no lugar dos circuitos de voo da medula e dos
músculos de direção):
  - altitude: ângulo de ataque das asas;
  - direção: diferença de ângulo de ataque entre as asas esquerda e direita;
  - voo para frente: o corpo inclina o nariz para baixo (halteres mantêm a
    inclinação) e a sustentação vira empuxo;
  - velocidade: inclinação do corpo (para frente acelera, para trás freia);
  - pouso: freia no ar e desce devagar até as patas tocarem o chão.
"""

from __future__ import annotations

import numpy as np

from mosca.cerebro import PASTA_DADOS

DOBRADA = 0.0  # rad: asa dobrada sobre o abdômen
RHO_AR = 1.2e-6  # g/mm^3
AREA_ASA = 2.0  # mm^2
CENTRO_PRESSAO = {"l": np.array([-0.13, 1.6, 0.15]), "r": np.array([-0.13, -1.6, 0.15])}  # frame da asa


class TabelaAsa:
    """(phi, alfa) -> ângulos (yaw, pitch, roll) das articulações da asa."""

    PHI = np.radians(np.arange(-100, 101, 4))
    ALFA = np.radians(np.arange(-84, 85, 4))

    def __init__(self, corpo):
        cache = PASTA_DADOS / "asa_cinematica_inversa.npz"
        if cache.exists():
            self.q = np.load(cache)["q"]
            return
        import mujoco as mj
        from scipy.optimize import least_squares

        m, d = corpo.sim.mj_model, corpo.sim.mj_data
        tid = corpo._torax_id
        qadr = [m.jnt_qposadr[m.joint(f"mosca/c_thorax-l_wing-{ax}").id] for ax in ("yaw", "pitch", "roll")]
        b = m.body("mosca/l_wing").id
        salvo = d.qpos.copy()

        def orientacao(q):
            d.qpos[qadr] = q
            mj.mj_kinematics(m, d)
            return d.xmat[tid].reshape(3, 3).T @ d.xmat[b].reshape(3, 3)

        self.q = np.zeros((len(self.PHI), len(self.ALFA), 3))
        q0 = np.zeros(3)
        for i, phi in enumerate(self.PHI):
            for j, alfa in enumerate(self.ALFA):
                alvo = orientacao_asa(phi, alfa)
                r = least_squares(lambda x: (orientacao(x) - alvo).ravel(), q0)
                self.q[i, j] = r.x
                q0 = r.x if j else r.x
        d.qpos[:] = salvo
        np.savez(cache, q=self.q)

    def __call__(self, phi: float, alfa: float) -> np.ndarray:
        x = np.interp(phi, self.PHI, np.arange(len(self.PHI)))
        y = np.interp(alfa, self.ALFA, np.arange(len(self.ALFA)))
        i, j = min(int(x), len(self.PHI) - 2), min(int(y), len(self.ALFA) - 2)
        u, v = x - i, y - j
        q = self.q
        return (q[i, j] * (1 - u) * (1 - v) + q[i + 1, j] * u * (1 - v) + q[i, j + 1] * (1 - u) * v + q[i + 1, j + 1] * u * v)


def orientacao_asa(phi: float, alfa: float) -> np.ndarray:
    """Orientação desejada da asa esquerda no frame do tórax (colunas: corda,
    envergadura, normal). phi > 0 = asa para a frente; alfa > 0 = bordo de
    ataque para cima em relação ao movimento para a frente."""
    s = np.array([np.sin(phi), np.cos(phi), 0.0])
    t = np.array([np.cos(phi), -np.sin(phi), 0.0])
    z = np.array([0.0, 0.0, 1.0])
    corda = np.cos(alfa) * t + np.sin(alfa) * z
    normal = -np.sin(alfa) * t + np.cos(alfa) * z
    return np.column_stack([corda, s, normal])


def coeficientes(alfa: float) -> tuple[float, float]:
    """Robofly (Dickinson et al. 1999), alfa em radianos (0 a pi/2)."""
    a = np.degrees(alfa)
    cl = 0.225 + 1.58 * np.sin(np.radians(2.13 * a - 7.2))
    cd = 1.92 - 1.55 * np.cos(np.radians(2.04 * a - 9.82))
    return max(cl, 0.0), max(cd, 0.0)


class Asas:
    def __init__(self, corpo, freq: float = 200.0, amp: float = np.radians(55), rot: float = np.radians(20)):
        self.c = corpo
        self.freq, self.amp0, self.rot = freq, amp, rot  # rot = ângulo de ataque no meio da batida
        self.tabela = TabelaAsa(corpo)
        self.fase = 0.0
        self.ativo = False
        self.pousando = False
        self.fim = 0.0
        self.t = 0.0
        self.direcao = np.array([1.0, 0.0])
        self.altura_alvo = 4.0  # mm acima do chão
        self.inclinacao = np.radians(8)  # nariz para baixo -> voo para frente (ajustada pela velocidade)
        self.velocidade_alvo = 300.0  # mm/s, voo de fuga
        self._pouso_inicio = None
        self._integral = np.zeros(2)
        self._cima = np.array([0.0, 0.0, 1.0])
        self.amp_atual = np.zeros(2)
        self.rot_atual = np.zeros(2)
        self.voos = 0

    # ---- comandos --------------------------------------------------------
    def decolar(self, duracao_s: float, direcao_mundo) -> None:
        d = np.asarray(direcao_mundo, dtype=float)[:2]
        self.direcao = d / (np.linalg.norm(d) + 1e-9)
        self.ativo, self.pousando = True, False
        self.fim = self.t + duracao_s
        self._integral[:] = 0
        self.voos += 1

    # ---- a cada passo de física -------------------------------------------
    def alvos(self, dt: float) -> np.ndarray:
        """Ângulos-alvo das articulações das asas (l: yaw, pitch, roll; r: idem)."""
        self.t += dt
        if not self.ativo:
            self.amp_atual[:] = 0
            return np.zeros(6)
        c = self.c
        d = c.sim.mj_data
        pos = d.xpos[c._torax_id]
        vz = c._vel_torax()[5]
        if self.t > self.fim and not self.pousando:
            self.pousando = True
            self._pouso_inicio = (self.t, pos[2])
        if self.pousando:  # freia no ar e desce devagar (~15 mm/s)
            t0, z0 = self._pouso_inicio
            alvo_z = max(0.6, z0 - 15.0 * max(self.t - t0 - 0.15, 0.0))
        else:
            alvo_z = self.altura_alvo
        rot = self.rot + 0.08 * (alvo_z - pos[2]) - 4e-4 * vz
        if self.pousando:
            if pos[2] < 1.25 or (pos[2] < 2.0 and self.t - self._pouso_inicio[0] > 0.3):  # patas no chão
                self._integral[:] = 0
                self.ativo = False
                self.rot_atual[:] = 0
                return np.zeros(6)
        # parede da arena: desvia a direção de voo para dentro
        raio = self.c.raio_arena
        if raio:
            r = np.hypot(pos[0], pos[1])
            if r > raio - 14:
                radial = pos[:2] / r
                para_fora = float(self.direcao @ radial)
                if para_fora > -0.3:
                    novo = self.direcao - (para_fora + 0.6) * radial
                    self.direcao = novo / np.linalg.norm(novo)
        # velocidade (como um drone): o corpo inclina na direção da aceleração
        # desejada; correção proporcional + integral (compensa desvios das asas)
        v_xy = self.c._vel_torax()[3:5]
        v_alvo = 0.0 if self.pousando else self.velocidade_alvo
        erro_v = v_alvo * self.direcao - v_xy
        self._integral = np.clip(self._integral + erro_v * dt, -150, 150)
        acel = 4.0 * erro_v + 6.0 * self._integral  # mm/s^2
        incl = acel / 9810.0
        n = np.linalg.norm(incl)
        if n > np.tan(np.radians(30)):
            incl *= np.tan(np.radians(30)) / n
        self._cima = np.array([incl[0], incl[1], 1.0]) / np.sqrt(1 + incl @ incl)
        # direção: gira para a direção desejada com diferença de ângulo de ataque
        frente = self.c.direcao()[:2]
        erro = np.arctan2(frente[0] * self.direcao[1] - frente[1] * self.direcao[0], frente @ self.direcao)
        dif = np.clip(0.3 * erro, -0.3, 0.3)
        self.rot_atual[:] = np.clip([rot * (1 - dif), rot * (1 + dif)], np.radians(5), np.radians(70))
        self.amp_atual[:] = self.amp0
        self.fase += 2 * np.pi * self.freq * dt
        phi = self.amp0 * np.sin(self.fase)
        virada = np.tanh(3 * np.cos(self.fase)) / np.tanh(3)  # alfa vira na inversão
        out = np.zeros(6)
        for k, lado in enumerate((0, 3)):
            out[lado : lado + 3] = self.tabela(phi, self.rot_atual[k] * virada)
        return out

    def forcas(self) -> np.ndarray:
        """Sustentação e arrasto das duas asas, devolvidos como força e torque
        no tórax (a batida é imposta pelos músculos; aplicar a força na asa de
        2,5 microgramas deixaria a integração numérica instável)."""
        import mujoco as mj

        c = self.c
        m, d = c.sim.mj_model, c.sim.mj_data
        vel = np.zeros(6)
        total = np.zeros(6)
        if not self.ativo:
            return total
        com = d.xipos[c._torax_id]
        for lado in "lr":
            b = c._asa_ids[lado]
            R = d.xmat[b].reshape(3, 3)
            cp = d.xpos[b] + R @ CENTRO_PRESSAO[lado]
            mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY, b, vel, 0)
            v = vel[3:] + np.cross(vel[:3], cp - d.xpos[b])
            u = np.linalg.norm(v)
            if u < 1.0:
                continue
            vu = v / u
            n = R[:, 2]  # normal da asa
            nv = float(n @ vu)
            alfa = np.arcsin(min(abs(nv), 1.0))
            cl, cd = coeficientes(alfa)
            q = 0.5 * RHO_AR * u * u * AREA_ASA
            perp = n - nv * vu
            pn = np.linalg.norm(perp)
            dir_l = -np.sign(nv) * perp / pn if pn > 1e-9 else np.zeros(3)
            f = q * (cl * dir_l - cd * vu)
            total[:3] += f
            total[3:] += np.cross(cp - com, f)
        # arrasto do corpo (tórax + abdômen ~ 1,5 mm2 de área frontal, Cd ~ 1)
        v_corpo = c._vel_torax()[3:]
        total[:3] += -0.5 * RHO_AR * 1.0 * 1.5 * np.linalg.norm(v_corpo) * v_corpo
        self.forca_atual = total
        return total

    def cima_desejado(self) -> np.ndarray:
        """Para onde o dorso deve apontar em voo (definido pelo controle de velocidade)."""
        return self._cima
