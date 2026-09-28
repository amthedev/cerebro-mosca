"""Corpo da mosca (NeuroMechFly v2 / FlyGym) num mundo com comida e predador.

O corpo tem as pernas controladas por um gerador de passos (CPG híbrido do
FlyGym, comandado por dois sinais descendentes: esquerda e direita) e a
probóscide articulada, para a mosca poder esticar a "língua" quando o MN9
dispara.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco as mj
import numpy as np
from flygym import Simulation
from flygym.anatomy import (
    PASSIVE_TARSAL_LINKS,
    ActuatedDOFPreset,
    AnatomicalJoint,
    AxisOrder,
    BodySegment,
    ContactBodiesPreset,
    JointPreset,
    Skeleton,
)
from flygym.compose import ActuatorType, FlatGroundWorld, KinematicPosePreset, NeuroMechFly
from flygym.utils.math import Rotation3D

from mosca.locomocao import ControladorPernas

# Juntas da probóscide (rostro e haustelo) que o MN9 comanda
PROBOSCIDE = [
    AnatomicalJoint("c_thorax", "c_head", []),  # cabeça fixa, só liga a árvore
    AnatomicalJoint("c_head", "c_rostrum", ["pitch"]),
    AnatomicalJoint("c_rostrum", "c_haustellum", ["pitch"]),
]
# ângulos (rad) somados à pose neutra quando a probóscide está estendida
EXTENSAO_PROBOSCIDE = np.array([-0.9, 0.9])


@dataclass
class Comida:
    x: float
    y: float
    raio: float = 2.5  # mm
    acucar: float = 1.0  # quantidade restante (0 a 1)
    alcance_cheiro: float = 14.0  # mm (desvio padrão da pluma)
    odores: tuple[str, ...] = ("C",)  # C = cheiro de comida fermentada
    esgotada_em: float | None = None


@dataclass
class FonteCheiro:
    """Fonte de cheiro sem comida (ver memoria.ODORES)."""

    x: float
    y: float
    odor: str
    alcance: float = 10.0  # mm
    intensidade: float = 1.0


CORES_ODOR = {"A": [0.2, 0.45, 1.0, 0.55], "B": [0.75, 0.2, 0.8, 0.55], "C": [1.0, 0.75, 0.1, 0.55]}


@dataclass
class Predador:
    """Objeto escuro que mergulha na mosca. Persegue a cabeça dela até 75% do
    ataque e então dá o bote no último ponto onde ela estava: se a mosca não
    saiu de lá a tempo, é pega."""

    inicio_s: float  # quando começa o ataque
    duracao_s: float = 0.6  # tempo até o bote
    raio: float = 1.5  # mm
    lado: str = "left"  # de que lado da mosca ele vem
    distancia_inicial: float = 40.0  # mm
    altura: float = 3.0  # mm
    raio_captura: float = 2.0  # mm
    pode_pegar: bool = True
    direcao_ataque: np.ndarray | None = None
    alvo: np.ndarray | None = None
    resultado: str = ""  # "", "pegou", "escapou"


class Corpo:
    def __init__(
        self,
        comidas: list[Comida],
        predadores: list[Predador] = (),
        *,
        fontes: list[FonteCheiro] = (),
        cheiro_ambiente: dict[str, float] | None = None,
        olhos: bool = False,
        raio_arena: float | None = None,
        voo: bool = False,
        renderizar: bool = True,
        resolucao=(360, 480),
        velocidade_video: float = 1.0,
        seed: int = 0,
    ):
        self.comidas = comidas
        self.predadores = list(predadores)
        self.fontes = list(fontes)
        self.cheiro_ambiente = dict(cheiro_ambiente or {})
        self.raio_arena = raio_arena

        pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(AxisOrder.YAW_PITCH_ROLL)
        juntas = JointPreset.LEGS_ONLY.to_joint_list() + PROBOSCIDE
        if voo:
            juntas += [AnatomicalJoint("c_thorax", f"{s}_wing", ["yaw", "pitch", "roll"]) for s in "lr"]
        esqueleto = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, anatomical_joints=juntas)
        fly = NeuroMechFly(name="mosca")
        for dof, junta in fly.add_joints(esqueleto, neutral_pose=pose, stiffness=0.05, damping=0.06).items():
            if dof.child.link in PASSIVE_TARSAL_LINKS:
                junta.stiffness[0] = 7.5
                junta.damping[0] = 1e-2
            if dof.child.name in ("l_wing", "r_wing"):  # asa: leve, sem mola, amortecida
                junta.stiffness[0] = 0.0
                junta.damping[0] = 0.04
                junta.armature = 1e-6
        dofs_pernas = esqueleto.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
        dofs_prob = [d for d in esqueleto.iter_jointdofs() if d.child.name in ("c_rostrum", "c_haustellum")]
        fly.add_actuators(dofs_pernas, ActuatorType.POSITION, neutral_input=pose, kp=45.0, forcerange=(-65.0, 65.0))
        fly.add_actuators(dofs_prob, ActuatorType.POSITION, neutral_input=pose, kp=5.0, forcerange=(-5.0, 5.0))
        dofs_asas = [d for d in esqueleto.iter_jointdofs() if d.child.name in ("l_wing", "r_wing")]
        if voo:
            # "músculo" da asa rígido (ressonância ~1 kHz, bem acima da batida de ~220 Hz)
            fly.add_actuators(dofs_asas, ActuatorType.POSITION, neutral_input=pose, kp=200.0, forcerange=(-2000.0, 2000.0))
            for g in fly.mjcf_root.geoms:  # o ar age numa superfície com a forma da asa
                if g.name in ("l_wing", "r_wing"):
                    g.fluid_ellipsoid = 0
            for lado, sy in (("l", 1), ("r", -1)):
                g = fly.mjcf_root.body(f"{lado}_wing").add_geom(
                    name=f"{lado}_asa_ar", type=mj.mjtGeom.mjGEOM_ELLIPSOID, size=[0.55, 1.15, 0.02],
                    pos=[-0.13, sy * 1.15, 0.15], mass=1e-8, contype=0, conaffinity=0, group=3,
                )  # fmt: skip
                g.fluid_ellipsoid = 0  # a aerodinâmica das asas vem de voo.Asas.forcas
        fly.add_leg_adhesion(gain=40.0)
        fly.colorize()
        if olhos:
            fly.add_vision()
        cam_lado = fly.add_tracking_camera(
            name="lado", pos_offset=(-0.3, -7.0, 1.2),
            rotation=Rotation3D("euler", (1.45, 0.0, 0.0)), fovy=40.0,
        )  # fmt: skip

        world = FlatGroundWorld()
        self._decorar_mundo(world.mjcf_root)
        world.add_fly(
            fly, [0, 0, 0.8], Rotation3D("quat", [1, 0, 0, 0]),
            bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
            add_ground_contact_sensors=False,
        )  # fmt: skip

        self.fly = fly
        self.sim = Simulation(world)
        self.dt = self.sim.timestep
        if voo:  # integrador estável para as asas leves
            self.sim.mj_model.opt.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
            self._asa_ids = {s: self.sim.mj_model.body(f"mosca/{s}_wing").id for s in "lr"}
        ordem = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
        self._idx_asas = np.array([ordem.index(d) for d in dofs_asas]) if voo else np.array([], dtype=int)
        self._idx_pernas = np.array([ordem.index(d) for d in dofs_pernas])
        self._idx_prob = np.array([ordem.index(d) for d in dofs_prob])
        self._prob_neutra = np.array([pose.joint_angles_lookup_rad.get(d.name, 0.0) for d in dofs_prob])
        self._acao = np.zeros(len(ordem))

        self.ctrl = ControladorPernas(self.sim, fly, dofs_pernas, seed=seed)
        segs = fly.get_bodysegs_order()
        self._i_torax = segs.index(BodySegment("c_thorax"))
        self._i_cabeca = segs.index(BodySegment("c_head"))
        self._i_boca = segs.index(BodySegment("c_haustellum"))
        self._i_antenas = [segs.index(BodySegment(f"{s}_funiculus")) for s in "lr"]
        self._torax_id = self.sim._internal_bodyids_by_fly[fly.name][self._i_torax]
        self._mocap = {
            nome: self.sim.mj_model.body(nome).mocapid[0]
            for nome in [f"predador{i}" for i in range(len(self.predadores))]
        }

        self.renderer = None
        if renderizar:
            self.renderer = self.sim.set_renderer(
                [cam_lado, "cima"], camera_res=resolucao,
                playback_speed=velocidade_video, output_fps=25,
            )  # fmt: skip

        self.sim.reset()
        self.ctrl.reset(seed=seed)
        self.extensao_prob = 0.0
        self._salto_passos = 0
        self._salto_forca = np.zeros(3)
        self._sem_adesao_passos = 0
        self._vel = np.zeros(6)
        self._na_parede = False
        self._historico_cabeca: list[tuple[float, np.ndarray]] = []
        self.olhos = None
        if olhos:
            from mosca.visao import Olhos

            self.olhos = Olhos(self)
        self.asas = None
        if voo:
            from mosca.voo import Asas

            self.asas = Asas(self)
        self._aplicar(self.ctrl.pose_neutra(), np.ones(6, bool))
        self.sim.warmup()

    # ---- mundo -----------------------------------------------------------
    def _decorar_mundo(self, spec: mj.MjSpec) -> None:
        corpo = spec.worldbody
        for i, c in enumerate(self.comidas):
            corpo.add_geom(
                name=f"comida{i}", type=mj.mjtGeom.mjGEOM_CYLINDER,
                pos=[c.x, c.y, 0.01], size=[c.raio, 0.01, 0],
                rgba=[1.0, 0.75, 0.1, 1.0], contype=0, conaffinity=0,
            )  # fmt: skip
        for i, f in enumerate(self.fontes):
            corpo.add_geom(
                name=f"fonte{i}", type=mj.mjtGeom.mjGEOM_CYLINDER,
                pos=[f.x, f.y, 0.01], size=[1.5, 0.01, 0],
                rgba=CORES_ODOR.get(f.odor, [0.6, 0.6, 0.6, 0.55]), contype=0, conaffinity=0,
            )  # fmt: skip
        for i, p in enumerate(self.predadores):
            b = corpo.add_body(name=f"predador{i}", mocap=True, pos=[0, 0, -50])
            b.add_geom(
                type=mj.mjtGeom.mjGEOM_SPHERE, size=[p.raio, 0, 0],
                rgba=[0.1, 0.1, 0.12, 1.0], contype=0, conaffinity=0,
            )  # fmt: skip
        if self.raio_arena:
            # parede da arena, baixa e clara (não parece sombra para os olhos);
            # a mosca sente e desvia (proximidade_parede) e a física a segura (passo)
            spec.add_material(name="parede", rgba=[0.95, 0.95, 0.93, 1.0], emission=0.8)
            n = 48
            for k in range(n):
                a = 2 * np.pi * k / n
                corpo.add_geom(
                    type=mj.mjtGeom.mjGEOM_BOX,
                    pos=[self.raio_arena * np.cos(a), self.raio_arena * np.sin(a), 0.6],
                    size=[0.4, np.pi * self.raio_arena / n + 0.05, 0.6],
                    quat=[np.cos(a / 2), 0, 0, np.sin(a / 2)],
                    material="parede", contype=0, conaffinity=0,
                )  # fmt: skip
            corpo.add_camera(name="cima", pos=[0, 0, self.raio_arena * 2.3], xyaxes=[1, 0, 0, 0, 1, 0], fovy=50)
            return
        xs = [0] + [c.x for c in self.comidas] + [f.x for f in self.fontes]
        ys = [0] + [c.y for c in self.comidas] + [f.y for f in self.fontes]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        altura = max(max(xs) - min(xs), max(ys) - min(ys)) * 1.2 + 45
        corpo.add_camera(name="cima", pos=[cx, cy, altura], xyaxes=[1, 0, 0, 0, 1, 0], fovy=50)

    # ---- leitura do corpo ------------------------------------------------
    def posicoes(self) -> np.ndarray:
        return self.sim.get_body_positions(self.fly.name)

    def torax(self) -> np.ndarray:
        return self.posicoes()[self._i_torax]

    def direcao(self) -> np.ndarray:
        """Vetor unitário para onde a mosca aponta (no plano do chão)."""
        v = self.sim.mj_data.xmat[self._torax_id].reshape(3, 3)[:, 0].copy()
        v[2] = 0
        return v / (np.linalg.norm(v) + 1e-9)

    def cheiros(self) -> dict[str, np.ndarray]:
        """Concentração (0-1) de cada cheiro na antena esquerda e direita."""
        pos = self.posicoes()[self._i_antenas]
        conc: dict[str, np.ndarray] = {o: np.full(2, x) for o, x in self.cheiro_ambiente.items()}

        def somar(odor, x, y, alcance, intensidade):
            d2 = (pos[:, 0] - x) ** 2 + (pos[:, 1] - y) ** 2
            conc[odor] = conc.get(odor, np.zeros(2)) + intensidade * np.exp(-d2 / (2 * alcance**2))

        for c in self.comidas:
            if c.esgotada_em is None:
                for o in c.odores:
                    somar(o, c.x, c.y, c.alcance_cheiro, c.acucar)
        for f in self.fontes:
            somar(f.odor, f.x, f.y, f.alcance, f.intensidade)
        return {o: np.clip(v, 0, 1) for o, v in conc.items()}

    def cheiro_nas_antenas(self) -> np.ndarray:
        """Concentração de cheiro de comida (C) na antena esquerda e direita."""
        return self.cheiros().get("C", np.zeros(2))

    def proximidade_parede(self) -> tuple[float, float]:
        """(0-1, lado): quão perto a cabeça está da parede e de que lado ela
        fica (+1 à esquerda, -1 à direita). Mecanossensores das antenas/pernas."""
        if not self.raio_arena:
            return 0.0, 0.0
        cab = self.posicoes()[self._i_cabeca]
        r = np.hypot(cab[0], cab[1])
        prox = float(np.clip(1 - (self.raio_arena - r) / 4.0, 0, 1))
        if prox <= 0:
            return 0.0, 0.0
        radial = np.array([cab[0], cab[1], 0.0]) / max(r, 1e-6)
        frente = self.direcao()
        esquerda = np.array([-frente[1], frente[0], 0.0])
        return prox, float(np.sign(radial @ esquerda) or 1.0)

    def atualizar_comidas(
        self, t: float, rng: np.random.Generator, espera_s: float = 25.0, longe_de=(0.0, 0.0), distancia_min: float = 18.0
    ) -> list[str]:
        """Comida esgotada some; depois de um tempo brota em outro lugar."""
        eventos = []
        m = self.sim.mj_model
        for i, c in enumerate(self.comidas):
            gid = m.geom(f"comida{i}").id
            if c.esgotada_em is None and c.acucar < 0.03:
                c.esgotada_em = t
                m.geom_rgba[gid][3] = 0.15
                eventos.append("comida acabou")
            elif c.esgotada_em is not None and t - c.esgotada_em > espera_s:
                raio = (self.raio_arena or 40.0) - 8
                while True:
                    x, y = rng.uniform(-raio, raio, 2)
                    cab = self.posicoes()[self._i_cabeca]
                    if (
                        np.hypot(x, y) < raio
                        and np.hypot(x - cab[0], y - cab[1]) > 15
                        and np.hypot(x - longe_de[0], y - longe_de[1]) > distancia_min
                    ):
                        break
                c.x, c.y, c.acucar, c.esgotada_em = float(x), float(y), 1.0, None
                m.geom_pos[gid][:2] = [x, y]
                m.geom_rgba[gid][3] = 1.0
                eventos.append("comida brotou")
        return eventos

    def comida_na_boca(self) -> tuple[float, Comida | None]:
        """Açúcar sob a boca (0 se a boca não está sobre comida)."""
        boca = self.posicoes()[self._i_boca]
        for c in self.comidas:
            if np.hypot(boca[0] - c.x, boca[1] - c.y) < c.raio and c.acucar > 0:
                return c.acucar, c
        return 0.0, None

    def mover_predadores(self, t: float, atraso_s: float = 0.12) -> list[str]:
        """Posiciona os predadores; devolve eventos ("pegou"/"escapou").
        O predador vê a mosca com atraso (tempo de reação ~120 ms)."""
        eventos = []
        agora = self.posicoes()[self._i_cabeca].copy()
        self._historico_cabeca.append((t, agora))
        while len(self._historico_cabeca) > 2 and self._historico_cabeca[1][0] <= t - atraso_s:
            self._historico_cabeca.pop(0)
        cabeca = self._historico_cabeca[0][1]
        frente = self.direcao()
        esquerda = np.array([-frente[1], frente[0], 0.0])
        for i, p in enumerate(self.predadores):
            pos = self._posicao_predador(p, t, cabeca, frente, esquerda)
            self.sim.mj_data.mocap_pos[self._mocap[f"predador{i}"]] = pos
            if p.alvo is not None and not p.resultado and t >= p.inicio_s + p.duracao_s:
                dist = np.hypot(*(agora[:2] - p.alvo[:2]))
                p.resultado = "pegou" if (p.pode_pegar and dist < p.raio_captura) else "escapou"
                eventos.append(p.resultado)
        return eventos

    def ameaca_visual(self, t: float) -> np.ndarray:
        """Sinal de looming (0-1) em cada olho pela geometria (sem retina):
        tamanho angular do predador. Usado quando o corpo não tem olhos."""
        sinal = np.zeros(2)
        cabeca = self.posicoes()[self._i_cabeca]
        frente = self.direcao()
        esquerda = np.array([-frente[1], frente[0], 0.0])
        for i, p in enumerate(self.predadores):
            if not (p.inicio_s <= t <= p.inicio_s + p.duracao_s):
                continue
            rel = self.sim.mj_data.mocap_pos[self._mocap[f"predador{i}"]] - cabeca
            d = np.linalg.norm(rel)
            angulo = 2 * np.arctan(p.raio / max(d, 1e-6))
            lado = 0 if rel @ esquerda > 0 else 1
            sinal[lado] = max(sinal[lado], np.clip((angulo - 0.1) / 0.6, 0, 1))
        return sinal

    def _posicao_predador(self, p, t, cabeca, frente, esquerda):
        fim = p.inicio_s + p.duracao_s
        if t < p.inicio_s or t > fim + 0.4:
            if t < p.inicio_s:
                p.direcao_ataque, p.alvo, p.resultado = None, None, ""
            return np.array([0, 0, -50.0])
        if p.direcao_ataque is None:  # direção de ataque fixa, vinda de um lado
            lado = esquerda if p.lado == "left" else -esquerda
            p.direcao_ataque = lado + 0.3 * frente
            p.direcao_ataque /= np.linalg.norm(p.direcao_ataque)
        u = np.clip((t - p.inicio_s) / p.duracao_s, 0, 1)
        if u < 0.75:  # perseguindo a cabeça
            d = p.distancia_inicial * (1 - u / 0.75) ** 2 + 3.0 * (u / 0.75)
            pos = cabeca + p.direcao_ataque * d
            pos[2] = p.altura + (cabeca[2] - p.altura) * (u / 0.75) * 0.5
            p.alvo = cabeca.copy()
            p._pos_bote = pos.copy()
            return pos
        if t <= fim:  # bote: vai direto ao último ponto onde a mosca estava
            w = (u - 0.75) / 0.25
            return p._pos_bote + (p.alvo - p._pos_bote) * w
        # depois do bote vai embora para cima
        return p.alvo + np.array([0, 0, 30.0 * (t - fim)])

    # ---- comandos --------------------------------------------------------
    def _aplicar(self, angulos_pernas, adesao):
        self._acao[self._idx_pernas] = angulos_pernas
        self._acao[self._idx_prob] = self._prob_neutra + EXTENSAO_PROBOSCIDE * self.extensao_prob
        if self.asas is not None:
            self._acao[self._idx_asas] = self.asas.alvos(self.dt)
        self.sim.set_actuator_inputs(self.fly.name, ActuatorType.POSITION, self._acao)
        self.sim.set_leg_adhesion_states(self.fly.name, adesao)

    def saltar(self, direcao_mundo: np.ndarray, velocidade: float = 250.0) -> None:
        """Salto de fuga (o que a Fibra Gigante faz: aciona o músculo de salto
        das pernas do meio). Impulso de 5 ms para cima e na direção dada."""
        if self._salto_passos > 0 or self._sem_adesao_passos > 0:
            return
        d = np.asarray(direcao_mundo, dtype=float)[:2]
        d = d / (np.linalg.norm(d) + 1e-9)
        v = velocidade * np.array([0.7 * d[0], 0.7 * d[1], 0.7])
        massa = float(self.sim.mj_model.body_subtreemass[self._torax_id])
        self._salto_passos = int(round(0.005 / self.dt))
        self._salto_forca = massa * v / 0.005
        self._sem_adesao_passos = int(round(0.12 / self.dt))

    def _vel_torax(self) -> np.ndarray:
        """[rot(3), lin(3)] do tórax no frame do mundo."""
        mj.mj_objectVelocity(self.sim.mj_model, self.sim.mj_data, mj.mjtObj.mjOBJ_BODY, self._torax_id, self._vel, 0)
        return self._vel

    @property
    def voando(self) -> bool:
        return self.asas is not None and self.asas.ativo

    def _torque_estabilizador(self) -> np.ndarray:
        """No ar, os halteres mantêm o corpo nivelado; no chão, se a mosca
        tombou (mais de 60 graus), ela se endireita (reflexo de endireitamento)."""
        m, d = self.sim.mj_model, self.sim.mj_data
        z = d.xmat[self._torax_id].reshape(3, 3)[:, 2]
        inclinacao = np.degrees(np.arccos(np.clip(z[2], -1, 1)))
        if not (self.saltando or inclinacao > 60 or self._na_parede or self.voando):
            return np.zeros(3)
        mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY, self._torax_id, self._vel, 0)
        w = self._vel[:3]
        cima = self.asas.cima_desejado() if self.voando else np.array([0.0, 0.0, 1.0])
        erro = np.cross(z, cima)  # eixo para girar até a atitude desejada
        if self.voando:  # halteres em voo; a guinada fica livre (a direção vem das asas)
            return 30.0 * erro - 0.15 * np.array([w[0], w[1], 0.1 * w[2]])
        return 8.0 * erro - 0.05 * w

    @property
    def saltando(self) -> bool:
        return self._sem_adesao_passos > 0

    def passo(self, sinal_descendente: np.ndarray, extensao_proboscide: float) -> None:
        """Avança a física um passo (0.1 ms)."""
        self.extensao_prob = extensao_proboscide
        if self.voando:  # em voo as pernas ficam paradas e soltas
            sinal_descendente = np.zeros(2)
            self._sem_adesao_passos = max(self._sem_adesao_passos, int(0.05 / self.dt))
        angulos, adesao = self.ctrl.step(sinal_descendente)
        if self._sem_adesao_passos > 0:
            adesao = np.zeros(6, dtype=bool)
            self._sem_adesao_passos -= 1
        self._aplicar(angulos, adesao)
        xfrc = self.sim.mj_data.xfrc_applied
        if self._salto_passos > 0:
            xfrc[self._torax_id, :3] = self._salto_forca
            self._salto_passos -= 1
        else:
            xfrc[self._torax_id, :3] = 0.0
        xfrc[self._torax_id, 3:] = self._torque_estabilizador()
        if self.asas is not None:
            xfrc[self._torax_id] += self.asas.forcas()
        self._na_parede = False
        if self.raio_arena:  # parede: segura quem tenta atravessar (mola + amortecedor)
            d = self.sim.mj_data
            pos = d.xpos[self._torax_id]
            r = np.hypot(pos[0], pos[1])
            invadiu = r - (self.raio_arena - 1.0)
            if invadiu > 0:
                self._na_parede = True
                radial = pos[:2] / r
                mj.mj_objectVelocity(self.sim.mj_model, d, mj.mjtObj.mjOBJ_BODY, self._torax_id, self._vel, 0)
                v_rad = float(self._vel[3:5] @ radial)
                massa = float(self.sim.mj_model.body_subtreemass[self._torax_id])
                xfrc[self._torax_id, :2] += -massa * (1.5e4 * invadiu + 150.0 * max(v_rad, 0.0)) * radial
        self.sim.step()
        if self.renderer is not None:
            self.sim.render_as_needed()
