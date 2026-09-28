"""Ciclo de vida: cérebro e corpo rodando juntos, com metabolismo e memória.

A cada bloco de 1 ms:
  corpo percebe o mundo -> neurônios sensoriais -> cérebro roda 1 ms ->
  corpo cogumelo aprende -> neurônios descendentes/motores -> comando ->
  corpo anda 1 ms -> energia gasta/ganha

O mesmo cérebro pode viver em vários mundos seguidos (sessões), guardando o
que aprendeu, como num protocolo de laboratório (treino -> teste).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from mosca import biologia
from mosca.cerebro import Cerebro
from mosca.corpo import Comida, Corpo, FonteCheiro, Predador
from mosca.estados import Relogio, Sono
from mosca.memoria import CorpoCogumelo
from mosca.neuronios import Catalogo
from mosca.ponte import Ponte, Sentidos


@dataclass
class Metabolismo:
    """Energia de 0 (morta de fome) a 1 (cheia)."""

    energia: float = 0.35
    gasto_basal: float = 0.003  # por segundo
    gasto_andar: float = 0.006  # por segundo, em velocidade normal
    gasto_voar: float = 0.04  # por segundo (voar custa ~5x mais que andar)
    ganho_comer: float = 0.10  # por segundo com a probóscide estendida no açúcar
    consumo_comida: float = 0.03  # açúcar da fonte consumido por segundo

    @property
    def fome(self) -> float:
        return 1.0 - self.energia


@dataclass
class Sessao:
    nome: str
    frames: dict = field(default_factory=dict)
    registro: list = field(default_factory=list)


class Vida:
    def __init__(
        self,
        comidas: list[Comida],
        predadores: list[Predador] = (),
        *,
        fontes: list[FonteCheiro] = (),
        cheiro_ambiente: dict | None = None,
        energia: float = 0.35,
        memoria: bool = False,
        olhos: bool = False,
        raio_arena: float | None = None,
        medula: bool = False,
        voo: bool = False,
        relogio: Relogio | None = None,
        sono: Sono | None = None,
        renderizar: bool = True,
        seed: int = 0,
        nome_sessao: str = "",
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.raio_arena = raio_arena
        self.relogio = relogio
        self.sono = sono
        self.t_vida = 0.0
        self.gravador = None
        self.titulo = ""
        self.renderizar = renderizar
        self.com_olhos = olhos
        self.com_voo = voo
        self.causa_morte = ""
        self.eventos: list[tuple[float, str]] = []
        self.cerebro = Cerebro(seed=seed)
        self.cat = Catalogo(self.cerebro.con.ids)
        self.mb = None
        if memoria:
            biologia.aplicar(self.cerebro, self.cat)
            self.mb = CorpoCogumelo(self.cerebro, self.cat, repouso_mbon=-46.0, eta=1e-4)
        self.acoplador = None
        if medula:
            from mosca.medula import AcopladorDescendentes, Medula

            self.acoplador = AcopladorDescendentes(self.cerebro, self.cat, Medula(seed=seed))
        self.meta = Metabolismo(energia=energia)
        self.viva = True
        self.sessoes: list[Sessao] = []
        self.corpo = None
        self.novo_mundo(comidas, predadores, fontes=fontes, cheiro_ambiente=cheiro_ambiente, nome=nome_sessao)

    def novo_mundo(self, comidas, predadores=(), *, fontes=(), cheiro_ambiente=None, nome="") -> None:
        """Coloca o mesmo cérebro (com o que aprendeu) num mundo novo."""
        self._fechar_sessao()
        self.corpo = Corpo(
            comidas, predadores, fontes=fontes, cheiro_ambiente=cheiro_ambiente,
            olhos=self.com_olhos, raio_arena=self.raio_arena, voo=self.com_voo,
            renderizar=self.renderizar, seed=self.seed,
        )  # fmt: skip
        self.ponte = Ponte(self.cerebro, self.cat, seed=self.seed, memoria=self.mb)
        self.ponte.salto_pela_medula = self.acoplador is not None
        self.ponte._contagem_ant = {k: int(self.cerebro.contagem[i].sum()) for k, i in self.ponte.grupos.items()}
        if self.mb is not None:
            self.mb.zerar_leitura()
        self.t = 0.0
        self.sessoes.append(Sessao(nome))

    @property
    def registro(self) -> list[dict]:
        return self.sessoes[-1].registro

    def _fechar_sessao(self) -> None:
        if self.sessoes and self.corpo is not None and self.corpo.renderer is not None:
            self.sessoes[-1].frames = self.corpo.renderer.frames

    def viver(self, duracao_s: float, bloco_ms: float = 1.0, mostrar_cada_s: float = 1.0, ao_vivo=None) -> None:
        """ao_vivo(vida, comando) é chamado a cada bloco (ex.: janela 3D); se
        devolver False, a sessão termina antes."""
        bloco_s = bloco_ms * 1e-3
        passos_corpo = int(round(bloco_s / self.corpo.dt))
        n_blocos = int(round(duracao_s / bloco_s))
        inicio = time.time()
        morreu_em = None
        for b in range(n_blocos):
            corpo, meta = self.corpo, self.meta
            luz = self.relogio.luz(self.t_vida) if self.relogio else 1.0
            if corpo.olhos is not None:
                corpo.olhos.luz = luz
            for ev in corpo.atualizar_comidas(self.t, self.rng):
                self.eventos.append((self.t_vida, ev))
            acucar, fonte = corpo.comida_na_boca()
            if corpo.voando:  # no ar não sente o gosto
                acucar, fonte = 0.0, None
            cheiros = corpo.cheiros()
            for ev in corpo.mover_predadores(self.t):
                self.eventos.append((self.t, ev))
                print(f"t={self.t:5.1f}s  predador: {ev.upper()}", flush=True)
                if ev == "pegou" and self.viva:
                    self.viva, self.causa_morte = False, "pega pelo predador"
            ameaca = corpo.olhos.atualizar(self.t) if corpo.olhos is not None else corpo.ameaca_visual(self.t)
            parede, parede_lado = corpo.proximidade_parede()
            s = Sentidos(
                acucar=acucar, cheiro=cheiros.get("C", np.zeros(2)),
                ameaca=ameaca, cheiros=cheiros, parede=parede, parede_lado=parede_lado,
            )  # fmt: skip

            dormindo = False
            if self.sono is not None and self.viva:
                mudou = self.sono.atualizar(bloco_s, luz, meta.fome)
                if self.ponte.taxa["GF"] > 30 and self.sono.acordar():
                    mudou = "acordou com susto"
                if mudou:
                    self.eventos.append((self.t_vida, mudou))
                    print(f"t={self.t:5.1f}s  sono: {mudou}", flush=True)
                dormindo = self.sono.dormindo
            pressao = self.sono.pressao if self.sono else 0.0
            self.ponte.sentir(s, meta.fome, pressao, dormindo)
            self.cerebro.rodar(bloco_ms)
            self.ponte.ler(bloco_ms)
            disparos_ttm = self.acoplador.passo(bloco_ms) if self.acoplador is not None else 0
            if self.mb is not None:
                self.mb.atualizar(bloco_ms)
            cmd = self.ponte.comandar(self.t, s, meta.fome, bloco_s, dormindo=dormindo)
            if not self.viva:
                cmd.sinal[:] = 0.0
                cmd.probóscide = 0.0
                cmd.modo = "MORTA"
                cmd.salto = None
            if disparos_ttm > 0 and self.viva and self.t >= getattr(self, "_proximo_salto", 0.0):
                cmd.salto = float(self.ponte._fuga_giro)  # TTMn da medula disparou: salto
            if cmd.salto is not None:
                frente = corpo.direcao()
                esquerda = np.array([-frente[1], frente[0], 0.0])
                corpo.saltar(frente - 1.5 * cmd.salto * esquerda)
                self.eventos.append((self.t, "salto"))
                self._proximo_salto = self.t + 1.0  # depois de decolar, não salta de novo logo
                self._direcao_fuga = frente - 1.5 * cmd.salto * esquerda
            # voo: os músculos de voo (DLM) disparam (Fibra Gigante -> PSI -> DLM)
            # ou, sem a medula, junto com o salto de fuga
            if corpo.asas is not None and self.viva and not corpo.voando:
                pelo_dlm = self.acoplador is not None and self.acoplador.disparos_voo > 0
                sem_medula = self.acoplador is None and cmd.salto is not None
                if pelo_dlm or sem_medula:
                    direcao = getattr(self, "_direcao_fuga", corpo.direcao())
                    corpo.asas.velocidade_alvo = 200.0
                    corpo.asas.decolar(0.5, direcao)
                    self.eventos.append((self.t, "voo"))
            if corpo.voando:
                cmd.modo = "VOANDO"
            for _ in range(passos_corpo):
                corpo.passo(cmd.sinal, cmd.probóscide)

            # metabolismo
            esforco = float(np.abs(cmd.sinal).mean())
            basal = meta.gasto_basal * (0.5 if dormindo else 1.0)
            gasto = meta.gasto_voar if corpo.voando else meta.gasto_andar * esforco
            meta.energia -= bloco_s * (basal + gasto)
            if fonte is not None and cmd.probóscide > 0.5:
                meta.energia += bloco_s * meta.ganho_comer * cmd.probóscide
                fonte.acucar = max(0.0, fonte.acucar - bloco_s * meta.consumo_comida)
            meta.energia = float(np.clip(meta.energia, 0.0, 1.0))
            if meta.energia <= 0 and self.viva:
                self.viva, self.causa_morte = False, "fome"

            if b % 10 == 0:
                self._registrar(s, cmd, luz)
            if self.gravador is not None and self.registro:
                self.gravador.talvez(self, self.registro[-1], self.titulo)
            self.t += bloco_s
            self.t_vida += bloco_s
            if ao_vivo is not None and ao_vivo(self, cmd) is False:
                break
            if not self.viva:
                morreu_em = self.t if morreu_em is None else morreu_em
                if self.t - morreu_em > 3.0:
                    print(f"t={self.t:5.1f}s  a mosca morreu ({self.causa_morte})", flush=True)
                    break
            if mostrar_cada_s and b % int(round(mostrar_cada_s / bloco_s)) == 0:
                r = self.ponte.taxa
                extra = ""
                if self.mb is not None:
                    extra = f" lembrança={self.ponte.valencia:+.2f} memória={self.mb.forca_memoria():.2%}"
                print(
                    f"t={self.t:5.1f}s  {cmd.modo:15s} energia={meta.energia:.2f} "
                    f"pos=({corpo.torax()[0]:5.1f},{corpo.torax()[1]:5.1f})  "
                    f"MN9={r['MN9']:4.0f}Hz GF={r['GF']:4.0f}Hz{extra} "
                    f"[{time.time() - inicio:4.0f}s de relógio]",
                    flush=True,
                )
        self._fechar_sessao()

    def _registrar(self, s: Sentidos, cmd, luz: float = 1.0) -> None:
        pos = self.corpo.torax()
        linha = dict(
            t=self.t, t_vida=self.t_vida, x=pos[0], y=pos[1], modo=cmd.modo, luz=luz,
            energia=self.meta.energia, acucar=s.acucar,
            cheiro_e=s.cheiro[0], cheiro_d=s.cheiro[1],
            ameaca_e=s.ameaca[0], ameaca_d=s.ameaca[1],
            sinal_e=cmd.sinal[0], sinal_d=cmd.sinal[1], proboscide=cmd.probóscide,
            **{f"hz_{k}": v for k, v in self.ponte.taxa.items()},
        )  # fmt: skip
        for o, c in s.cheiros.items():
            linha[f"odor_{o}"] = float(c.mean())
        if self.mb is not None:
            linha["valencia"] = self.ponte.valencia
            linha["memoria"] = self.mb.forca_memoria()
            linha["dopa_pam"] = float(self.cerebro.taxa_estimulo[self.mb.pam].mean())
            linha["dopa_ppl1"] = float(self.cerebro.taxa_estimulo[self.mb.ppl1].mean())
        if self.sono is not None:
            linha["sono"] = self.sono.pressao
        self.registro.append(linha)
