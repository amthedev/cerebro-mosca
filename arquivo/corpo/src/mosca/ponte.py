"""Ponte entre corpo e cérebro.

Sentidos -> cérebro: converte o que o corpo percebe no mundo em disparos dos
neurônios sensoriais reais do conectoma.
Cérebro -> corpo: lê neurônios descendentes/motores e decide o comando das
pernas e da probóscide.

Cada canal diz de onde vem: "conectoma" (o comportamento sai da fiação real)
ou "atalho" (regra nossa, provisória, até ser trocada por biologia).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mosca.cerebro import Cerebro
from mosca.memoria import CorpoCogumelo
from mosca.neuronios import ACUCAR_SHIU, Catalogo

CANAIS = {
    "paladar (açúcar -> GRNs)": "conectoma",
    "olfato (cheiro -> ORNs)": "conectoma",
    "ameaça visual (-> LPLC2)": "atalho: olho geométrico, sem retina",
    "comer (MN9 -> probóscide)": "conectoma",
    "fuga (Fibra Gigante DNp01)": "conectoma",
    "virar na fuga (DNa01/DNa02)": "conectoma",
    "andar para trás (MDN)": "conectoma",
    "andar para frente (DNp09)": "conectoma",
    "exploração pela fome": "atalho: estado interno",
    "virar para o cheiro": "atalho: comparação das antenas",
    "fome aumenta o gosto do açúcar": "atalho: modulação por dopamina",
    "cheiro -> PNs (lobo antenal idealizado)": "atalho",
    "código do cheiro nas células de Kenyon": "conectoma",
    "aprendizado KC->MBON pela dopamina": "conectoma + regra de plasticidade",
    "açúcar/ameaça -> dopamina (PAM/PPL1)": "atalho",
    "lembrança do cheiro -> aproximar/evitar": "atalho: leitura dos MBONs",
}


@dataclass
class Sentidos:
    acucar: float = 0.0  # 0-1, açúcar sob a boca
    cheiro: np.ndarray = field(default_factory=lambda: np.zeros(2))  # comida (C), antena E, D
    ameaca: np.ndarray = field(default_factory=lambda: np.zeros(2))  # olho E, D
    cheiros: dict = field(default_factory=dict)  # odor -> concentração nas antenas E, D
    parede: float = 0.0  # 0-1, quão perto da parede
    parede_lado: float = 0.0  # +1 parede à esquerda, -1 à direita


@dataclass
class Comando:
    sinal: np.ndarray  # descendente esquerdo, direito (CPG)
    probóscide: float  # 0 recolhida, 1 estendida
    modo: str
    salto: float | None = None  # salto de fuga agora; valor = giro (>0 para a direita)


class Ponte:
    # constante de tempo (ms) do filtro de cada grupo: fuga é rápida, comer é lento
    TAU_MS = {"MN9": 120.0, "GF": 20.0, "DNa_E": 40.0, "DNa_D": 40.0, "MDN": 60.0, "DNp09": 60.0}

    def __init__(self, cerebro: Cerebro, cat: Catalogo, seed: int = 0, memoria: CorpoCogumelo | None = None):
        self.c = cerebro
        self.mb = memoria
        self.valencia = 0.0
        self.salto_pela_medula = False  # se True, quem dispara o salto é o TTMn da medula
        self.rng = np.random.default_rng(seed)
        # entradas
        self.in_acucar = cerebro.idx(ACUCAR_SHIU)
        self.in_cheiro = [cerebro.idx(cat.olfato_comida(s)) for s in ("left", "right")]
        self.in_lplc2 = [cerebro.idx(cat.lplc2(s)) for s in ("left", "right")]
        # sono: ER5 acumulam pressão de sono; dFB (FB6*) promovem o sono
        self.in_er5 = cerebro.idx(cat.tipo("ER5"))
        self.in_dfb = cerebro.idx(cat.ids(cell_type=lambda t: t.startswith("FB6")))
        # saídas: grupos de neurônios lidos como taxa média
        self.grupos = {
            "MN9": cerebro.idx(cat.mn9()),
            "GF": cerebro.idx(cat.tipo("DNp01")),
            "DNa_E": cerebro.idx(cat.tipo("DNa01", "left") + cat.tipo("DNa02", "left")),
            "DNa_D": cerebro.idx(cat.tipo("DNa01", "right") + cat.tipo("DNa02", "right")),
            "MDN": cerebro.idx(cat.tipo("MDN")),
            "DNp09": cerebro.idx(cat.tipo("DNp09")),
        }
        self.taxa = {k: 0.0 for k in self.grupos}
        self._contagem_ant = {k: 0 for k in self.grupos}
        self._fuga_ate = -1.0
        self._fuga_giro = 0.0
        self._ruido_giro = 0.0
        # linha de base lenta dos DNa: a fuga usa só a mudança causada pela
        # ameaça (o cheiro sozinho já deixa os DNa esquerdos ativos no modelo)
        self._base_dna = np.zeros(2)
        self._comendo = False
        self._susto = 0.0  # traço de punição depois de uma ameaça
        self._v_rapida = 0.0  # lembrança filtrada (~100 ms)
        self._v_lenta = 0.0  # lembrança filtrada (~600 ms)
        self._manobra_ate = -1.0
        self._manobra_giro = 0.0

    # ---- sentidos -> cérebro --------------------------------------------
    def sentir(self, s: Sentidos, fome: float, pressao_sono: float = 0.0, dormindo: bool = False) -> None:
        self.c.estimular_idx(self.in_er5, 40.0 * pressao_sono)
        self.c.estimular_idx(self.in_dfb, 30.0 if dormindo else 0.0)
        if dormindo:  # dormindo, o limiar de despertar sobe: sentidos químicos atenuados
            s = Sentidos(acucar=0.2 * s.acucar, cheiro=0.3 * s.cheiro, ameaca=s.ameaca,
                         cheiros={o: 0.3 * c for o, c in s.cheiros.items()})  # fmt: skip
        # 300 Hz porque no conectoma o cheiro de comida inibe o MN9: com cheiro
        # presente é preciso mais açúcar (medido em experimentos/04_calibrar.py)
        ganho_acucar = np.clip(1.2 * fome, 0.0, 1.0)
        self.c.estimular_idx(self.in_acucar, 300.0 * s.acucar * ganho_acucar)
        for lado in range(2):
            if self.mb is None:  # sem memória: cheiro de comida entra pelos ORNs
                self.c.estimular_idx(self.in_cheiro[lado], 50.0 * s.cheiro[lado] if s.cheiro[lado] > 0.02 else 0.0)
            self.c.estimular_idx(self.in_lplc2[lado], 150.0 * s.ameaca[lado])
        if self.mb is not None:
            # cheiro -> neurônios olfativos de cada antena -> lobo antenal
            self.mb.cheirar({o: c for o, c in s.cheiros.items() if c.max() > 0.02})
            # reforço: açúcar na boca -> PAM (recompensa); ameaça -> PPL1
            # (punição), que continua ~1,5 s depois do susto
            self._susto *= np.exp(-1e-3 / 1.5)
            if s.ameaca.max() > 0.3:
                self._susto = 1.0
            punicao = max(float(np.clip(2 * (s.ameaca.max() - 0.15), 0, 1)), self._susto)
            self.mb.reforcar(recompensa=float(s.acucar > 0), punicao=punicao)

    def ler(self, janela_ms: float) -> None:
        """Atualiza as taxas filtradas dos grupos de saída."""
        for k, idx in self.grupos.items():
            a = np.exp(-janela_ms / self.TAU_MS[k])
            total = int(self.c.contagem[idx].sum())
            hz = (total - self._contagem_ant[k]) / len(idx) / (janela_ms * 1e-3)
            self._contagem_ant[k] = total
            self.taxa[k] = a * self.taxa[k] + (1 - a) * hz

    # ---- cérebro -> corpo -----------------------------------------------
    def comandar(self, t: float, s: Sentidos, fome: float, dt: float, dormindo: bool = False) -> Comando:
        r = self.taxa
        if self.mb is not None:
            self.valencia = self.mb.valencia_aprendida()
        dna = np.array([r["DNa_E"], r["DNa_D"]])
        extra = dna - self._base_dna
        # fuga: a Fibra Gigante dispara -> corrida rápida, virando para o lado
        # que os DNa indicam (longe da ameaça)
        if r["GF"] > 30 and t > self._fuga_ate:
            self._fuga_ate = t + 0.6
            self._fuga_giro = np.clip((extra[1] - extra[0]) / 40.0, -0.5, 0.5)
            # a Fibra Gigante aciona o salto (músculo de salto das pernas do meio);
            # com a medula ligada, o salto vem do motoneurônio TTMn (vida.py)
            salto = None if self.salto_pela_medula else float(self._fuga_giro)
            return Comando(self._lados(1.5, self._fuga_giro), 0.0, "FUGA", salto=salto)
        if t < self._fuga_ate:
            return Comando(self._lados(1.5, self._fuga_giro), 0.0, "FUGA")
        a = np.exp(-dt / 1.0)
        self._base_dna = a * self._base_dna + (1 - a) * dna
        if dormindo:
            return Comando(np.zeros(2), 0.0, "DORMINDO")

        # comer: MN9 ativo -> para de andar e estende a probóscide
        # (histerese: começa acima de 20 Hz, só para abaixo de 8 Hz)
        self._comendo = r["MN9"] > (8.0 if self._comendo else 20.0)
        if self._comendo:
            return Comando(np.zeros(2), float(np.clip(r["MN9"] / 40.0, 0, 1)), "COMENDO")

        if r["MDN"] > 20:
            return Comando(np.full(2, -0.8), 0.0, "RÉ")

        # andar: exploração (fome) + DNp09, virando para o cheiro + ruído
        self._ruido_giro += (-self._ruido_giro / 0.5) * dt + 0.5 * np.sqrt(dt) * self.rng.standard_normal()
        c = s.cheiro
        giro_cheiro = 0.0
        modo = "EXPLORANDO"
        base = 0.35 + 0.8 * fome + r["DNp09"] / 50.0
        # inato: com fome, vira para o cheiro de comida
        if c.sum() > 0.02 and fome > 0.3:
            giro_cheiro = np.clip(40.0 * (c[1] - c[0]) / (c.sum() + 1e-9), -0.8, 0.8)
            base += 0.3 if c.sum() > 0.05 else 0.0
            modo = "SEGUINDO CHEIRO"
        # aprendido: klinotaxia pela lembrança do cheiro (saída dos MBONs).
        # Se a lembrança está melhorando, segue reto; se está piorando, vira.
        # Só usa o que o cérebro informa (não sabe de onde vem cada cheiro).
        if self.mb is not None:
            v = self.valencia
            a1, a2 = np.exp(-dt / 0.1), np.exp(-dt / 0.6)
            self._v_rapida = a1 * self._v_rapida + (1 - a1) * v
            self._v_lenta = a2 * self._v_lenta + (1 - a2) * v
            tendencia = self._v_rapida - self._v_lenta
            aprendidos = [x for o, x in s.cheiros.items() if o != "C"]
            # com fome e sentindo cheiro de comida, o instinto de ir até ela manda
            seguindo_comida = modo == "SEGUINDO CHEIRO"
            if aprendidos and max(x.mean() for x in aprendidos) > 0.05 and abs(self._v_rapida) > 0.05 and not seguindo_comida:
                base += 0.3
                modo = "LEMBRANÇA BOA" if self._v_rapida > 0 else "EVITANDO CHEIRO"
                if t < self._manobra_ate:
                    giro_cheiro = self._manobra_giro
                elif tendencia < -0.02 and t > self._manobra_ate + 0.3:
                    self._manobra_ate = t + 0.35
                    self._manobra_giro = 0.8 * self.rng.choice([-1.0, 1.0])
                    giro_cheiro = self._manobra_giro
                else:
                    giro_cheiro += 1e-3  # segue reto (pouco ruído)
                if self._v_rapida > 0.8 and max(x.mean() for x in aprendidos) > 0.7:
                    # chegou onde a lembrança é boa: busca local (devagar, girando)
                    base *= 0.35
                    giro_cheiro = 2.0 * self._ruido_giro
                    modo = "BUSCA LOCAL"
        if s.parede > 0:  # sente a parede e vira para o outro lado
            giro_cheiro += 1.5 * s.parede * s.parede_lado
        giro = np.clip(giro_cheiro + self._ruido_giro * (0.3 if giro_cheiro else 1.0), -0.9, 0.9)
        return Comando(self._lados(base, giro), 0.0, modo)

    @staticmethod
    def _lados(base: float, giro: float) -> np.ndarray:
        """giro > 0 vira para a direita (pernas esquerdas mais rápidas)."""
        return np.array([base * (1 + giro), base * (1 - giro)])
