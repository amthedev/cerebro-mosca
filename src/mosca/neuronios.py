"""Catálogo de neurônios por nome, a partir das anotações do FlyWire.

Fonte: Schlegel et al. 2024 (github.com/flyconnectome/flywire_annotations),
com ids da versão 783, a mesma do conectoma usado pelo cérebro.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd

from mosca.cerebro import PASTA_DADOS

ARQUIVO_ANOTACOES = PASTA_DADOS / "flywire_neuron_annotations.tsv"

# Glomérulos ativados por cheiro de comida fermentada (vinagre/fruta) e por
# cheiros de perigo. Semmelhack & Wang 2009; Stensmyr et al. 2012; Suh et al. 2004.
GLOMERULOS_COMIDA = ("DM1", "DM2", "DM4", "DP1m", "VA2", "VM2")
GLOMERULOS_PERIGO = ("V", "DA2")  # CO2, geosmina

# Neurônios gustativos de açúcar da probóscide usados por Shiu et al. 2024
# (tutorial do repositório), validados no experimento 01. Um id não existe na v783.
ACUCAR_SHIU = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]  # fmt: skip


class Catalogo:
    def __init__(self, ids_modelo: np.ndarray | None = None):
        ann = pd.read_csv(ARQUIVO_ANOTACOES, sep="\t", low_memory=False)
        if ids_modelo is not None:
            ann = ann[ann.root_id.isin(ids_modelo)]
        self.ann = ann.set_index("root_id")

    def ids(self, lado: str | None = None, **filtros) -> list[int]:
        """Ids por coluna de anotação. Valores podem ser str, tupla ou callable."""
        m = np.ones(len(self.ann), dtype=bool)
        for coluna, valor in filtros.items():
            col = self.ann[coluna].fillna("").astype(str)
            if callable(valor):
                m &= col.map(valor).to_numpy(dtype=bool)
            elif isinstance(valor, (tuple, list, set)):
                m &= col.isin(valor).to_numpy()
            else:
                m &= (col == valor).to_numpy()
        if lado is not None:
            m &= (self.ann.side == lado).to_numpy()
        return self.ann.index[m].tolist()

    # ---- sensoriais -----------------------------------------------------
    def acucar(self, lado=None):
        return self.ids(lado, cell_sub_class="sugar/water", cell_type="LB3")

    def amargo(self, lado=None):
        return self.ids(lado, cell_sub_class="bitter")

    def orn(self, glomerulos, lado=None):
        return self.ids(lado, cell_type=tuple(f"ORN_{g}" for g in glomerulos))

    def olfato_comida(self, lado=None):
        return self.orn(GLOMERULOS_COMIDA, lado)

    def olfato_perigo(self, lado=None):
        return self.orn(GLOMERULOS_PERIGO, lado)

    def lplc2(self, lado=None):
        """Detectores de objeto se aproximando (looming)."""
        return self.ids(lado, cell_type="LPLC2")

    def orgao_johnston(self, lado=None):
        """Mecanossensores da antena (vento, som, toque)."""
        return self.ids(lado, cell_type=lambda t: t.startswith("JO-"))

    # ---- saídas (neurônios descendentes e motores) ----------------------
    @cached_property
    def descendentes(self) -> list[int]:
        return self.ids(super_class="descending")

    def tipo(self, cell_type, lado=None):
        return self.ids(lado, cell_type=cell_type)

    def mn9(self, lado=None):
        """Motoneurônio que estende a probóscide (Shiu et al.)."""
        return self.ids(lado, cell_type="CB0701")

    def nome(self, root_id: int) -> str:
        if root_id not in self.ann.index:
            return str(root_id)
        linha = self.ann.loc[root_id]
        return f"{linha.cell_type}_{str(linha.side)[0].upper()}"
