"""Vida 3.0: um dia na vida da mosca.

Arena redonda com duas fontes de comida (cheiro de comida C + cheiro A) que
acabam e brotam em outro lugar, e uma zona de perigo com cheiro B onde
predadores atacam com frequência. O dia tem manhã, tarde, anoitecer e noite:
à noite os olhos quase não enxergam e a mosca dorme quando a pressão de sono
sobe. Ela aprende sozinha, pela dopamina, que A = comida e B = perigo.

    uv run python experimentos/09_um_dia.py                 # ~4 min de vida
    uv run python experimentos/09_um_dia.py --duracao 30 --dia 12 --noite 10   # teste rápido
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mosca import mundos
from mosca.cerebro import RAIZ
from mosca.mundos import UmDia
from mosca.video import CORES_MODO, GravadorVideo

p = argparse.ArgumentParser()
p.add_argument("--duracao", type=float, default=240.0)
p.add_argument("--dia", type=float, default=150.0, help="duração do dia (s)")
p.add_argument("--noite", type=float, default=60.0, help="duração da noite (s)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--sem-video", action="store_true")
p.add_argument("--sem-medula", action="store_true", help="salto pela regra da ponte, sem a medula BANC")
p.add_argument("--nome", default="um_dia")
args = p.parse_args()
pasta = RAIZ / "resultados"
pasta.mkdir(exist_ok=True)

mundo = UmDia(dia_s=args.dia, noite_s=args.noite, seed=args.seed, medula=not args.sem_medula)
vida = mundo.vida
RAIO, PERIGO = mundos.RAIO, mundos.PERIGO
COMIDAS_INICIAIS = mundos.COMIDAS_INICIAIS
if not args.sem_video:
    vida.gravador = GravadorVideo(pasta / f"{args.nome}.mp4", velocidade=4.0)

vida.viver(args.duracao, mostrar_cada_s=5.0, ao_vivo=mundo)
if vida.gravador is not None:
    print("vídeo:", vida.gravador.fechar())

reg = pd.DataFrame(vida.registro)
reg.to_csv(pasta / f"{args.nome}_registro.csv", index=False)
with open(pasta / f"{args.nome}_eventos.json", "w") as f:
    json.dump(vida.eventos, f)

# ---- resumo ----------------------------------------------------------------
duracao_real = reg.t_vida.iloc[-1]
tempo = reg.groupby("modo").size() / len(reg)
print(f"\nViveu {duracao_real:.0f} s ({'VIVA' if vida.viva else 'morreu: ' + vida.causa_morte})")
print("Tempo em cada comportamento:", ", ".join(f"{m} {v:.0%}" for m, v in tempo.sort_values(ascending=False).items()))
cont = pd.Series([e for _, e in vida.eventos]).value_counts()
print("Eventos:", ", ".join(f"{k}: {v}" for k, v in cont.items()))

fig = plt.figure(figsize=(12, 7))
g = fig.add_gridspec(3, 2, width_ratios=[2.2, 1], height_ratios=[0.6, 1, 1])
ax0 = fig.add_subplot(g[0, 0])
modos = list(dict.fromkeys(reg.modo))
for m in modos:
    sel = reg.modo == m
    ax0.scatter(reg.t_vida[sel], np.zeros(sel.sum()), marker="|", s=400,
                color=np.array(CORES_MODO.get(m, (90, 90, 90))) / 255, label=m)  # fmt: skip
ax0.set_yticks([])
ax0.set_title("Comportamento ao longo do dia")
ax0.legend(ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.25))
ax1 = fig.add_subplot(g[1, 0], sharex=ax0)
ax1.fill_between(reg.t_vida, 0, 1, where=reg.luz < 0.5, color="#1b2340", alpha=0.18, label="noite")
ax1.plot(reg.t_vida, reg.energia, color="#3a9a3a", label="energia")
ax1.plot(reg.t_vida, reg.sono, color="#4650a0", label="pressão de sono")
ax1.plot(reg.t_vida, reg.luz, color="#e0a020", lw=1, label="luz")
for t, e in vida.eventos:
    if e.startswith("ataque"):
        ax1.axvline(t, color="#d03030", lw=0.8, alpha=0.6)
ax1.set_ylim(0, 1.05)
ax1.legend(fontsize=7, loc="upper right")
ax1.set_title("Estado interno (linhas vermelhas = ataques de predador)")
ax2 = fig.add_subplot(g[2, 0], sharex=ax0)
ax2.plot(reg.t_vida, reg.valencia, color="#1a8f5a", label="lembrança do cheiro atual")
ax2.plot(reg.t_vida, reg.memoria * 20, color="#888888", label="sinapses KC->MBON mudadas (x20)")
ax2.axhline(0, color="k", lw=0.5)
ax2.legend(fontsize=7)
ax2.set_xlabel("tempo de vida (s)")
ax2.set_title("Memória")
ax3 = fig.add_subplot(g[:, 1])
ax3.add_patch(plt.Circle((0, 0), RAIO, fill=False, color="#8a7f70", lw=3))
ax3.add_patch(plt.Circle((PERIGO.x, PERIGO.y), 12, color="#b040c0", alpha=0.2))
ax3.plot(PERIGO.x, PERIGO.y, "o", color="#b040c0", ms=9, label="zona de perigo (B)")
for c in COMIDAS_INICIAIS:
    ax3.plot(*c, "o", color="#e0a020", ms=9)
ax3.plot(reg.x, reg.y, color="#333333", lw=0.7, alpha=0.8)
ax3.plot(reg.x.iloc[0], reg.y.iloc[0], "k^", ms=8)
ax3.set_aspect("equal")
ax3.set_title("Caminho (laranja = comida inicial)")
fig.tight_layout()
fig.savefig(pasta / f"{args.nome}_resumo.png", dpi=130)
print("resumo:", pasta / f"{args.nome}_resumo.png")
