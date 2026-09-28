"""Checagem do consenso de neurotransmissor por tipo no arquivo do BANC.

`banc_888_neurotransmitter_prediction_v2.csv` traz, para cada neurônio, a
previsão própria (`neurotransmitter_predicted`) e um consenso do seu tipo
(`cell_type_neurotransmitter_predicted`). Em vários tipos grandes o consenso
contradiz quase todos os neurônios do próprio tipo (ex.: células de Kenyon,
>99% acetilcolina, com consenso serotonina ou glutamato).
"""

import pandas as pd

from mosca.banc import PASTA_BANC

nt = pd.read_csv(PASTA_BANC / "banc_888_neurotransmitter_prediction_v2.csv", low_memory=False)
t = nt.dropna(subset=["cell_type"])
g = t.groupby("cell_type").agg(
    neuronios=("root_id", "size"),
    maioria=("neurotransmitter_predicted", lambda s: s.mode().iloc[0]),
    fracao_maioria=("neurotransmitter_predicted", lambda s: s.value_counts(normalize=True).iloc[0]),
    consenso=("cell_type_neurotransmitter_predicted", lambda s: s.mode().iloc[0] if s.notna().any() else None),
    confianca_consenso=("cell_type_neurotransmitter_score", "mean"),
)
g = g[g.neuronios >= 5]
fortes = g[g.fracao_maioria >= 0.9]
ruins = fortes[fortes.consenso.notna() & (fortes.consenso != fortes.maioria)].sort_values("neuronios", ascending=False)
print(f"tipos com >= 5 neurônios: {len(g)}; consenso = maioria dos neurônios em {(g.consenso == g.maioria).mean():.1%}")
print(f"tipos em que >= 90% dos neurônios concordam: {len(fortes)}; o consenso contradiz {len(ruins)} deles "
      f"({ruins.neuronios.sum()} neurônios)")  # fmt: skip
print(ruins.head(20).round(3).to_string())
ruins.to_csv(PASTA_BANC.parents[1] / "resultados" / "banc_consenso_nt_suspeito.csv")
