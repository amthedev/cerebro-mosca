"""Monta o vídeo final: as duas câmeras + um painel com sentidos, neurônios,
energia e memória. Várias sessões viram um vídeo só, com cartões de título."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

VERDE = (60, 170, 90)  # vem do conectoma
LARANJA = (230, 140, 40)  # atalho
CINZA = (90, 90, 90)
CORES_MODO = {
    "EXPLORANDO": (70, 110, 200),
    "SEGUINDO CHEIRO": (150, 90, 200),
    "LEMBRANÇA BOA": (40, 150, 110),
    "EVITANDO CHEIRO": (170, 60, 150),
    "BUSCA LOCAL": (30, 120, 90),
    "COMENDO": (230, 160, 20),
    "FUGA": (210, 50, 50),
    "VOANDO": (40, 160, 200),
    "RÉ": (120, 120, 120),
    "DORMINDO": (40, 50, 110),
    "MORTA": (0, 0, 0),
}
PAINEL_H = 232  # 360 + 232 = 592, divisível por 16 (codec)


def _fonte(tam):
    for nome in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]:
        try:
            return ImageFont.truetype(nome, tam)
        except OSError:
            pass
    return ImageFont.load_default()


F, FG, FP, FT = _fonte(15), _fonte(26), _fonte(12), _fonte(34)


def _barra(d, x, y, larg, valor, maximo, cor, rotulo, texto_valor, f):
    d.text((x, y), rotulo, fill=(30, 30, 30), font=f)
    y2 = y + 18
    d.rectangle([x, y2, x + larg, y2 + 10], fill=(225, 225, 225))
    frac = float(np.clip(valor / maximo, 0, 1))
    d.rectangle([x, y2, x + int(larg * frac), y2 + 10], fill=cor)
    d.text((x + larg + 6, y2 - 3), texto_valor, fill=(30, 30, 30), font=f)


def _barra_central(d, x, y, larg, valor, rotulo):
    """Barra de -1 a +1 com zero no meio (verde = atrai, roxo = repele)."""
    d.text((x, y), rotulo, fill=(30, 30, 30), font=FP)
    y2 = y + 18
    d.rectangle([x, y2, x + larg, y2 + 10], fill=(225, 225, 225))
    meio = x + larg // 2
    v = float(np.clip(valor, -1, 1))
    cor = (40, 150, 110) if v >= 0 else (170, 60, 150)
    d.rectangle([min(meio, meio + int(v * larg / 2)), y2, max(meio, meio + int(v * larg / 2)), y2 + 10], fill=cor)
    d.line([meio, y2 - 2, meio, y2 + 12], fill=(80, 80, 80))
    d.text((x + larg + 6, y2 - 3), f"{v:+.2f}", fill=(30, 30, 30), font=FP)


def _quadro(imagens: list[np.ndarray], linha, t: float, titulo: str = "") -> np.ndarray:
    h, w = imagens[0].shape[:2]
    img = Image.new("RGB", (w * len(imagens), h + PAINEL_H), (250, 250, 248))
    for k, im in enumerate(imagens):
        img.paste(Image.fromarray(im), (k * w, 0))
    d = ImageDraw.Draw(img)
    d.text((10, 8), f"{titulo}   t = {t:4.1f} s" if titulo else f"t = {t:4.1f} s", fill=(20, 20, 20), font=F)

    y0 = h + 10
    cor = CORES_MODO.get(linha.modo, CINZA)
    larg = max(200, int(d.textlength(linha.modo, font=FG)) + 26)
    d.rounded_rectangle([10, y0, 10 + larg, y0 + 44], radius=8, fill=cor)
    d.text((22, y0 + 8), linha.modo, fill=(255, 255, 255), font=FG)
    _barra(d, 10, y0 + 54, 170, linha.energia, 1.0, (60, 160, 60), "Energia", f"{linha.energia:.0%}", F)
    if "valencia" in linha and not pd.isna(linha.valencia):
        _barra_central(d, 10, y0 + 96, 170, linha.valencia, "Lembrança do cheiro (repele <-> atrai)")
        dopa = []
        if linha.dopa_pam > 5:
            dopa.append("recompensa (PAM)")
        if linha.dopa_ppl1 > 5:
            dopa.append("punição (PPL1)")
        d.text((10, y0 + 134), "Dopamina: " + (", ".join(dopa) if dopa else "-"), fill=(30, 30, 30), font=FP)
        d.text((10, y0 + 152), f"Sinapses KC->MBON mudadas: {linha.memoria:.2%}", fill=(30, 30, 30), font=FP)
    if "sono" in linha and not pd.isna(linha.sono):
        _barra(d, 10, y0 + 170, 170, linha.sono, 1.0, (70, 80, 160), "Pressão de sono", f"{linha.sono:.0%}", FP)

    total_w = w * len(imagens)
    col1, col2 = 330, 330 + (total_w - 330) // 2
    d.text((col1, y0), "SENTIDOS -> CÉREBRO", fill=(30, 30, 30), font=F)
    _barra(d, col1, y0 + 22, 150, linha.acucar, 1, VERDE, "Açúcar na boca", "", FP)
    odores = sorted(k[5:] for k in linha.index if k.startswith("odor_") and not pd.isna(linha[k]))
    if odores:
        texto = "  ".join(f"{o} {linha['odor_' + o]:.2f}" for o in odores)
        maior = max(linha["odor_" + o] for o in odores)
    else:
        texto, maior = f"E {linha.cheiro_e:.2f} / D {linha.cheiro_d:.2f}", max(linha.cheiro_e, linha.cheiro_d)
    _barra(d, col1, y0 + 58, 150, maior, 1, VERDE, f"Cheiros ({texto})", "", FP)
    _barra(d, col1, y0 + 94, 150, max(linha.ameaca_e, linha.ameaca_d), 1, LARANJA,
           "Ameaça vindo " + ("da esquerda" if linha.ameaca_e >= linha.ameaca_d else "da direita"), "", FP)  # fmt: skip

    d.text((col2, y0), "CÉREBRO -> CORPO (Hz)", fill=(30, 30, 30), font=F)
    itens = [
        ("MN9 - comer", linha.hz_MN9),
        ("Fibra Gigante - fugir", linha.hz_GF),
        ("DNa esquerda - virar", linha.hz_DNa_E),
        ("DNa direita - virar", linha.hz_DNa_D),
        ("MDN - dar ré", linha.hz_MDN),
    ]
    for j, (rot, v) in enumerate(itens):
        _barra(d, col2, y0 + 22 + 36 * j, 150, v, 150, VERDE, rot, f"{v:.0f}", FP)
    d.text((10, h + PAINEL_H - 20), "verde = sai do conectoma real   laranja = atalho provisório",
           fill=(110, 110, 110), font=FP)  # fmt: skip
    return np.asarray(img)


def _cartao(largura: int, altura: int, titulo: str, subtitulo: str) -> np.ndarray:
    img = Image.new("RGB", (largura, altura), (24, 26, 32))
    d = ImageDraw.Draw(img)
    tw = d.textlength(titulo, font=FT)
    d.text(((largura - tw) / 2, altura / 2 - 50), titulo, fill=(255, 255, 255), font=FT)
    for k, linha in enumerate(subtitulo.split("\n")):
        sw = d.textlength(linha, font=F)
        d.text(((largura - sw) / 2, altura / 2 + 10 + 24 * k), linha, fill=(200, 200, 210), font=F)
    return np.asarray(img)


def _quadros_sessao(frames: dict, registro: list[dict], fps: int, titulo: str = ""):
    reg = pd.DataFrame(registro)
    cams = list(frames.keys())
    n = min(len(frames[c]) for c in cams)
    for i in range(n):
        t = i / fps
        linha = reg.iloc[int(np.clip(np.searchsorted(reg.t.values, t), 0, len(reg) - 1))]
        yield _quadro([frames[c][i] for c in cams], linha, t, titulo)


def montar_video(frames: dict, registro: list[dict], saida: Path, fps: int = 25) -> Path:
    iio.imwrite(saida, np.stack(list(_quadros_sessao(frames, registro, fps))), fps=fps, codec="libx264", quality=7)
    return saida


def montar_sessoes(sessoes, saida: Path, fps: int = 25, legendas: dict | None = None) -> Path:
    """Junta várias sessões (vida.sessoes) com um cartão de título antes de cada uma."""
    legendas = legendas or {}
    todos = []
    for s in sessoes:
        quadros = list(_quadros_sessao(s.frames, s.registro, fps, s.nome))
        if not quadros:
            continue
        h, w = quadros[0].shape[:2]
        cartao = _cartao(w, h, s.nome, legendas.get(s.nome, ""))
        todos.extend([cartao] * int(2.0 * fps))
        todos.extend(quadros)
    iio.imwrite(saida, np.stack(todos), fps=fps, codec="libx264", quality=7)
    return saida


class GravadorVideo:
    """Grava o vídeo durante a vida (sem guardar tudo na memória), para vidas longas.

    velocidade = quantas vezes mais rápido que o tempo simulado (4 -> 1 s de
    vídeo mostra 4 s de vida)."""

    def __init__(self, caminho: Path, velocidade: float = 4.0, fps: int = 25, resolucao=(360, 480)):
        import imageio.v2 as iio2

        self.caminho = Path(caminho)
        self.escritor = iio2.get_writer(self.caminho, fps=fps, codec="libx264", quality=7, macro_block_size=16)
        self.intervalo = velocidade / fps
        self.resolucao = resolucao
        self.proximo = 0.0
        self._modelo = None
        self.n_quadros = 0

    def _renderizador(self, corpo):
        import mujoco as mj

        if self._modelo is not corpo.sim.mj_model:
            self._modelo = corpo.sim.mj_model
            h, w = self.resolucao
            self._rend = mj.Renderer(self._modelo, h, w)
            self._cams = [self._modelo.camera(n).id for n in ("mosca/lado", "cima")]
        return self._rend

    def talvez(self, vida, linha: dict, titulo: str = "") -> None:
        if vida.t_vida < self.proximo:
            return
        self.proximo = vida.t_vida + self.intervalo
        corpo = vida.corpo
        rend = self._renderizador(corpo)
        luz = float(linha.get("luz", 1.0))
        imagens = []
        for cam in self._cams:
            rend.update_scene(corpo.sim.mj_data, camera=cam)
            img = rend.render().astype(np.float32)
            escurece = 0.25 + 0.75 * luz  # noite: tudo mais escuro e azulado
            img = img * escurece + (1 - escurece) * np.array([10, 15, 45])
            imagens.append(np.clip(img, 0, 255).astype(np.uint8))
        if corpo.olhos is not None and corpo.olhos.ultima_imagem is not None:
            olho = Image.fromarray(corpo.olhos.imagem_legivel(0)).convert("RGB").resize((120, 120))
            base = Image.fromarray(imagens[0])
            base.paste(olho, (imagens[0].shape[1] - 126, 6))
            ImageDraw.Draw(base).text((imagens[0].shape[1] - 124, 128), "olho esquerdo", fill=(255, 255, 255), font=FP)
            imagens[0] = np.asarray(base)
        self.escritor.append_data(_quadro(imagens, pd.Series(linha), vida.t_vida, titulo))
        self.n_quadros += 1

    def fechar(self) -> Path:
        self.escritor.close()
        return self.caminho
