"""Figuras do artigo (em inglês, para o preprint), a partir dos experimentos 12-15.

Saída: resultados/figuras/fig1_motor.png ... fig4_duas_moscas.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mosca.cerebro import RAIZ

RES = RAIZ / "resultados"
FIG = RES / "figuras"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
AZUL, LARANJA, CINZA, VERDE = "#2b6cb0", "#dd6b20", "#a0aec0", "#2f855a"

# 1. motor: mesma tentativa do artigo (açúcar 150 Hz, 1 s simulado, 1 núcleo do M4)
fig, ax = plt.subplots(figsize=(4.2, 2.4))
nomes = ["original Brian2 code\n(rebuilds network\nevery trial)", "this work, dense", "this work, event-driven"]
tempos = [967.0, 1.59 + 0.4, 0.48 + 0.4]
ax.barh(nomes, tempos, color=[CINZA, AZUL, AZUL])
ax.set_xscale("log")
ax.set_xlabel("wall-clock seconds per 1-s trial, one M4 core")
for y, t in enumerate(tempos):
    ax.text(t * 1.15, y, f"{t:.1f} s", va="center")
ax.set_xlim(0.3, 5000)
ax.set_title("Identical spikes, ~1,000x faster", loc="left")
fig.tight_layout()
fig.savefig(FIG / "fig1_motor.png")

# 2. auditoria
aud = pd.read_csv(RES / "auditoria.csv")
conds = list(dict.fromkeys(aud["condição"]))
COND_EN = {"original": "original", "só moduladores": "modulators only", "só sinais": "signs only", "só histamina": "histamine only", "só lobo antenal": "antennal lobe only", "todas": "all corrections"}
rot = [COND_EN[c] for c in conds]
fig, axs = plt.subplots(1, 2, figsize=(7.5, 2.8))
kc = aud[aud.teste == "células de Kenyon ativas (%)"].set_index("condição").valor.astype(float)[conds]
axs[0].bar(range(len(conds)), kc, color=[LARANJA if v > 15 else VERDE for v in kc])
axs[0].axhspan(5, 10, color=VERDE, alpha=0.2)
axs[0].text(len(conds) - 0.5, 11, "real fly ~5-10%", ha="right", va="bottom", fontsize=7, color=VERDE)
axs[0].set_xticks(range(len(conds)), rot, rotation=35, ha="right")
axs[0].set_ylabel("Kenyon cells active (%)")
axs[0].set_title("A  Odor -> mushroom body", loc="left")
lb3 = aud[aud.teste == "LB3 esq / dir: neurônios ativos"].set_index("condição").valor[conds]
esq = [int(v.split("/")[0]) for v in lb3]
dirt = [int(v.split("/")[1]) for v in lb3]
x = np.arange(len(conds))
axs[1].bar(x - 0.2, esq, 0.4, color=AZUL, label="left sugar GRNs")
axs[1].bar(x + 0.2, dirt, 0.4, color=LARANJA, label="right sugar GRNs")
axs[1].set_yscale("log")
axs[1].set_xticks(x, rot, rotation=35, ha="right")
axs[1].set_ylabel("active neurons")
axs[1].set_ylim(200, 40000)
axs[1].set_title("B  Sugar: left vs right", loc="left")
axs[1].legend(frameon=False, fontsize=7, ncol=2, loc="upper center")
fig.tight_layout()
fig.savefig(FIG / "fig2_auditoria.png")

# 3. varredura do peso por sinapse
esc = pd.read_csv(RES / "escala.csv")
fig, axs = plt.subplots(1, 3, figsize=(8, 2.6), sharey=True)
for ax, (prot, g) in zip(axs, esc.groupby("protocolo", sort=False)):
    for mosca, cor in (("FlyWire", AZUL), ("BANC", LARANJA)):
        h = g[g.mosca == mosca]
        ax.plot(h.fator, h["neurônios ativos"], "o-", color=cor, label=mosca)
    ax.set_yscale("log")
    ax.set_xlabel("weight per synapse (x 0.275 mV)")
    ax.set_title({"açúcar": "sugar (LB3)", "ameaça": "looming (LPLC2)", "antena": "antenna (JO-C/E)"}[prot], loc="left")
axs[0].set_ylabel("active neurons")
axs[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / "fig3_escala.png")

# 4. duas moscas
dm = pd.read_csv(RES / "duas_moscas_resumo.csv")
dm["grupo"] = dm["comparação"].str.replace(r" \(.*\)", "", regex=True).str.replace(r" (FlyWire|BANC)$", "", regex=True)
dm.loc[dm["comparação"] == "hemisférios FlyWire", "grupo"] = "hemisférios (FlyWire)"
dm = dm[dm["comparação"] != "hemisférios BANC"]
versoes = list(dict.fromkeys(dm.versao))
grupos = ["ruído", "hemisférios (FlyWire)", "indivíduos"]
media = dm.groupby(["versao", "grupo"]).spearman.mean().unstack()[grupos].loc[versoes]
fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.5, 1]})
x = np.arange(len(versoes))
for k, (g, cor) in enumerate(zip(grupos, (CINZA, VERDE, LARANJA))):
    rotulo = {"ruído": "same fly, other seed", "hemisférios (FlyWire)": "left vs right hemisphere", "indivíduos": "FlyWire vs BANC"}[g]
    axs[0].bar(x + (k - 1) * 0.27, media[g], 0.27, color=cor, label=rotulo)
VERSAO_EN = {
    "como está": "v2\nas is",
    "escala global": "v2\nglobal scale",
    "calibrado": "v2\nper-type",
    "calibrado + consenso BANC": "v2 per-type\n+ BANC type NT",
    "calibrado + sinais do FlyWire": "v2 per-type\n+ FlyWire NT",
    "v3 como está": "v3\nas is",
    "v3 calibrado + consenso BANC": "v3 per-type\n+ BANC type NT",
}
axs[0].set_xticks(x, [VERSAO_EN[v] for v in versoes], fontsize=5.5)
axs[0].set_xlabel("BANC version (synapse table, calibration, transmitter labels)", fontsize=7)
axs[0].set_ylabel("Spearman ρ across cell types")
axs[0].set_title("A  Agreement (mean of 6 stimuli)", loc="left", pad=22)
axs[0].legend(frameon=False, fontsize=7, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
axs[0].axhline(0, color="k", lw=0.5)
said = dm[dm["comparação"] == "indivíduos (esq)"].dropna(subset=["saida_fw"])
prots = list(dict.fromkeys(said.protocolo))
for k, v in enumerate(versoes):
    h = said[said.versao == v].set_index("protocolo").loc[prots]
    w = 0.8 / len(versoes)
    axs[1].bar(np.arange(len(prots)) + (k - (len(versoes) - 1) / 2) * w, h.saida_banc, w, label="BANC " + VERSAO_EN[v].replace("\n", " "))
fw_saida = said[said.versao == versoes[0]].set_index("protocolo").loc[prots].saida_fw
axs[1].scatter(np.arange(len(prots)), fw_saida, marker="_", s=400, color="k", zorder=3, label="FlyWire")
axs[1].set_xticks(range(len(prots)), ["sugar -> MN9", "LPLC2 -> giant fiber", "JO-C/E -> aBN1"], fontsize=7)
axs[1].set_ylabel("firing rate (Hz)")
axs[1].set_title("B  Key outputs", loc="left", pad=22)
axs[1].legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
fig.tight_layout()
fig.savefig(FIG / "fig4_duas_moscas.png")
print("figuras em", FIG)
