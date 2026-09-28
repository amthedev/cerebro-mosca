"""Estados internos de longo prazo: relógio (dia/noite) e sono.

Sono (modelo de dois processos, Borbély 1982, adaptado para moscas):
  - processo S: pressão de sono sobe enquanto acordada e cai enquanto dorme;
  - processo C: o relógio circadiano soma uma vontade de dormir forte à noite
    e tira um pouco de dia;
  - dorme quando S + C passa de um limiar e só acorda quando S + C cai bem
    abaixo dele (histerese). À noite C sozinho já mantém o sono, então a mosca
    dorme até o amanhecer mesmo depois de um cochilo à tarde; acorda antes só
    com fome forte ou com susto (Fibra Gigante).
No cérebro: a pressão ativa os neurônios ER5 (acumulam pressão de sono, Liu et
al. 2016) e dormir ativa os neurônios dFB (FB6*, promovem sono, Donlea et al.).
O desligamento do movimento durante o sono é atalho (ponte.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Relogio:
    """Dia comprimido: duração total, com amanhecer/anoitecer suaves."""

    duracao_dia_s: float = 150.0
    duracao_noite_s: float = 60.0
    transicao_s: float = 12.0
    luz_noite: float = 0.03

    @property
    def ciclo_s(self) -> float:
        return self.duracao_dia_s + self.duracao_noite_s

    def fase(self, t: float) -> float:
        """Posição no ciclo: 0 = começo do amanhecer. A vida começa de manhã."""
        return (t + self.transicao_s) % self.ciclo_s

    def luz(self, t: float) -> float:
        """1 = dia claro, luz_noite = noite; amanhecer e anoitecer suaves."""
        x, d, tr = self.fase(t), self.duracao_dia_s, self.transicao_s
        if x < tr:
            f = 0.5 * (1 - np.cos(np.pi * x / tr))
        elif x < d - tr:
            f = 1.0
        elif x < d:
            f = 0.5 * (1 + np.cos(np.pi * (x - (d - tr)) / tr))
        else:
            f = 0.0
        return self.luz_noite + (1 - self.luz_noite) * f

    def e_noite(self, t: float) -> bool:
        return self.luz(t) < 0.5


@dataclass
class Sono:
    pressao: float = 0.1  # processo S: 0 descansada, 1 exausta
    dormindo: bool = False
    sobe_por_s: float = 1 / 200  # acordada
    desce_por_s: float = 1 / 60  # dormindo
    c_noite: float = 0.45  # processo C à noite
    c_dia: float = -0.10  # processo C de dia
    limiar_dormir: float = 0.6
    limiar_acordar: float = 0.25
    fome_acorda: float = 0.85

    def circadiano(self, luz: float) -> float:
        return self.c_noite + (self.c_dia - self.c_noite) * float(np.clip(luz, 0, 1))

    def vontade(self, luz: float) -> float:
        return self.pressao + self.circadiano(luz)

    def atualizar(self, dt: float, luz: float, fome: float) -> str | None:
        """Devolve "dormiu"/"acordou"/"acordou com fome" quando muda de estado."""
        if self.dormindo:
            self.pressao = max(0.0, self.pressao - self.desce_por_s * dt)
            if fome > self.fome_acorda:
                self.dormindo = False
                return "acordou com fome"
            if self.vontade(luz) < self.limiar_acordar:
                self.dormindo = False
                return "acordou"
        else:
            self.pressao = min(1.0, self.pressao + self.sobe_por_s * dt)
            if self.vontade(luz) > self.limiar_dormir and fome < 0.7:
                self.dormindo = True
                return "dormiu"
        return None

    def acordar(self) -> bool:
        if self.dormindo:
            self.dormindo = False
            return True
        return False
