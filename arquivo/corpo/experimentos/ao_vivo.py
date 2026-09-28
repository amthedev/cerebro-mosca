"""Assistir a mosca viva numa janela 3D (visualizador do MuJoCo).

    uv run mjpython experimentos/ao_vivo.py                     # aprende e depois é testada
    uv run mjpython experimentos/ao_vivo.py --cenario simples   # Vida 1.0: comida e predador
    uv run mjpython experimentos/ao_vivo.py --cenario dia       # Vida 3.0: um dia inteiro

Mouse: botão esquerdo gira, botão direito arrasta, rodinha aproxima.
A simulação roda ~3x mais devagar que o tempo real (cérebro inteiro + física).
"""

import argparse
import time

import mujoco
import mujoco.viewer

from mosca.corpo import Comida, FonteCheiro, Predador
from mosca.vida import Vida

p = argparse.ArgumentParser()
p.add_argument("--cenario", choices=["aprender", "simples", "dia"], default="aprender")
p.add_argument("--duracao", type=float, default=240.0, help="só para o cenário dia")
args = p.parse_args()

ROTULOS = "Sessão\nTempo\nModo\nEnergia\nLembrança do cheiro\nMN9 (comer)\nFibra Gigante (fugir)\nDNa esq. / dir. (virar)\nPressão de sono"


def assistir(vida: Vida, duracao: float, titulo: str, mundo=None) -> bool:
    """Roda uma sessão com a janela aberta. Devolve False se a janela foi fechada."""
    m, d = vida.corpo.sim.mj_model, vida.corpo.sim.mj_data
    fechou = False
    with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as janela:
        janela.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        janela.cam.trackbodyid = int(vida.corpo._torax_id)
        janela.cam.distance, janela.cam.elevation, janela.cam.azimuth = 16.0, -35.0, 125.0
        ultimo = [0.0]
        luzes = (m.vis.headlight.ambient.copy(), m.vis.headlight.diffuse.copy(), m.light_diffuse.copy(), m.light_ambient.copy())

        def ao_vivo(vida, cmd):
            nonlocal fechou
            if not janela.is_running():
                fechou = True
                return False
            if mundo is not None:
                mundo(vida, cmd)
            agora = time.time()
            if agora - ultimo[0] > 1 / 30:  # ~30 quadros por segundo de relógio
                ultimo[0] = agora
                if vida.relogio is not None:  # noite: a cena escurece
                    f = 0.12 + 0.88 * vida.relogio.luz(vida.t_vida)
                    m.vis.headlight.ambient[:] = luzes[0] * f
                    m.vis.headlight.diffuse[:] = luzes[1] * f
                    m.light_diffuse[:] = luzes[2] * f
                    m.light_ambient[:] = luzes[3] * f
                r = vida.ponte.taxa
                sono = f"{vida.sono.pressao:.0%}" if vida.sono else "-"
                valores = (
                    f"{vida.titulo or titulo}\n{vida.t:.1f} s\n{cmd.modo}\n{vida.meta.energia:.0%}\n"
                    f"{vida.ponte.valencia:+.2f}\n{r['MN9']:.0f} Hz\n{r['GF']:.0f} Hz\n"
                    f"{r['DNa_E']:.0f} / {r['DNa_D']:.0f} Hz\n{sono}"
                )
                janela.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, ROTULOS, valores))
                janela.sync()
            return True

        vida.viver(duracao, ao_vivo=ao_vivo)
    return not fechou


if args.cenario == "dia":
    from mosca.mundos import UmDia

    mundo = UmDia()
    assistir(mundo.vida, args.duracao, "Um dia", mundo=mundo)
elif args.cenario == "simples":
    vida = Vida([Comida(14.0, 6.0)], [Predador(inicio_s=8.5, lado="left")], renderizar=False)
    assistir(vida, 12.0, "Vida 1.0")
else:
    vida = Vida([Comida(14.0, 6.0, odores=("C", "A"))], memoria=True, energia=0.35, renderizar=False)
    if assistir(vida, 9.0, "1. comida com cheiro A"):
        vida.novo_mundo(
            [], [Predador(inicio_s=t, lado=lado) for t, lado in [(1.0, "left"), (2.6, "right"), (4.2, "left")]],
            cheiro_ambiente={"B": 0.8},
        )  # fmt: skip
        if assistir(vida, 5.5, "2. cheiro B + predador"):
            vida.meta.energia = 0.5
            vida.novo_mundo([], fontes=[FonteCheiro(16.0, 9.0, "A"), FonteCheiro(16.0, -9.0, "B")])
            assistir(vida, 15.0, "3. teste: A (azul) ou B (roxo)?")
