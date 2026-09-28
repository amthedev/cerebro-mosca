"""Vida 1.0: a mosca com fome sente o cheiro, acha a comida, come até ficar
satisfeita e foge quando um predador se aproxima."""

import argparse
from pathlib import Path

import pandas as pd

from mosca.cerebro import RAIZ
from mosca.corpo import Comida, Predador
from mosca.video import montar_video
from mosca.vida import Vida

p = argparse.ArgumentParser()
p.add_argument("--duracao", type=float, default=12.0)
p.add_argument("--predador", type=float, default=8.5, help="segundo em que o predador ataca")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

vida = Vida(
    comidas=[Comida(x=14.0, y=6.0)],
    predadores=[Predador(inicio_s=args.predador, lado="left")],
    energia=0.35,
    seed=args.seed,
)
vida.viver(args.duracao)

pasta = RAIZ / "resultados"
pasta.mkdir(exist_ok=True)
pd.DataFrame(vida.registro).to_csv(pasta / "vida_registro.csv", index=False)
video = montar_video(vida.corpo.renderer.frames, vida.registro, pasta / "vida.mp4")
print("vídeo:", video)
