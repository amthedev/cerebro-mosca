"""Mundos prontos para a mosca viver.

UmDia: arena redonda com duas fontes de comida (cheiro de comida C + cheiro A),
que acabam e brotam em outro lugar, e uma zona de perigo no meio (cheiro B)
onde predadores atacam com frequência. Tem manhã, tarde, anoitecer e noite.
"""

from __future__ import annotations

import numpy as np

from mosca.corpo import Comida, FonteCheiro, Predador
from mosca.estados import Relogio, Sono
from mosca.vida import Vida

RAIO = 45.0
COMIDAS_INICIAIS = [(28.0, 6.0), (-28.0, -6.0)]
PERIGO = FonteCheiro(0.0, 0.0, "B", alcance=10.0)  # no meio, entre as comidas


class UmDia:
    def __init__(self, dia_s: float = 150.0, noite_s: float = 60.0, seed: int = 0,
                 medula: bool = True, voo: bool = True, renderizar: bool = False):  # fmt: skip
        self.relogio = Relogio(duracao_dia_s=dia_s, duracao_noite_s=noite_s)
        comidas = [Comida(x, y, odores=("C", "A")) for x, y in COMIDAS_INICIAIS]
        self.vida = Vida(
            comidas, [Predador(inicio_s=1e9)], fontes=[PERIGO],
            memoria=True, olhos=True, raio_arena=RAIO, relogio=self.relogio, sono=Sono(),
            medula=medula, voo=voo, energia=0.55, renderizar=renderizar, seed=seed,
        )  # fmt: skip
        self.predador = self.vida.corpo.predadores[0]
        self._proximo_ataque = 5.0

    def hora(self, t: float) -> str:
        """Dia (com amanhecer e anoitecer) = 05:00 às 20:00; noite = 20:00 às 05:00."""
        r = self.relogio
        x = r.fase(t)
        if x < r.duracao_dia_s:
            h = 5 + 15 * x / r.duracao_dia_s
        else:
            h = (20 + 9 * (x - r.duracao_dia_s) / r.duracao_noite_s) % 24
        dia = int((t + r.transicao_s) // r.ciclo_s) + 1
        return f"Dia {dia}, {int(h):02d}:{int(60 * (h % 1)):02d}"

    def __call__(self, v: Vida, cmd) -> bool:
        """Eventos do mundo a cada ms: relógio no título e ataques de predador
        (só de dia, muito mais frequentes perto do cheiro B)."""
        v.titulo = self.hora(v.t_vida) + ("  (noite)" if self.relogio.e_noite(v.t_vida) else "")
        p, t = self.predador, v.t
        atacando = p.inicio_s <= t <= p.inicio_s + p.duracao_s + 0.5
        if atacando or not v.viva or self.relogio.luz(v.t_vida) < 0.5 or t < self._proximo_ataque:
            return True
        cab = v.corpo.torax()
        perto = np.hypot(cab[0] - PERIGO.x, cab[1] - PERIGO.y) < 12
        taxa = 1.0 if perto else 1 / 90  # ataques por segundo
        if v.rng.random() < taxa * 1e-3:
            p.inicio_s = t + 0.001
            p.direcao_ataque, p.alvo, p.resultado = None, None, ""
            p.lado = str(v.rng.choice(["left", "right"]))
            self._proximo_ataque = t + 4.0
            v.eventos.append((v.t_vida, "ataque" + (" (zona B)" if perto else "")))
            print(f"t={t:5.1f}s  PREDADOR ATACA{' na zona B' if perto else ''}", flush=True)
        return True
