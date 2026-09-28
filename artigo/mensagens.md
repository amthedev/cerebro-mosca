# Rascunhos de mensagens para as equipes (você revisa e envia)

O canal certo para as três equipes é abrir uma *issue* (um relato público) no
GitHub de cada projeto. É assim que cientistas de dados abertos preferem
receber avisos de problemas. As issues são públicas, então revise o texto antes
de enviar. Confira também se o seu repositório já está no ar, porque os textos
apontam para ele.

Tom: educado, específico, reproduzível e sem acusar ninguém. Todo conjunto de
dados grande tem erros, e as equipes costumam agradecer relatos assim.

---

## 1. Modelo de Shiu et al. — github.com/philshiu/Drosophila_brain_model/issues

**O que diz, em português:** um neurônio inibitório conhecido (il3LN6) está
como excitatório no arquivo do modelo, e isso faz o açúcar do lado direito
"explodir" o cérebro. Há outros ~2.600 sinais que contrariam a literatura, e o
sistema olfativo do modelo fica ativo demais. As previsões validadas do artigo
continuam de pé.

**Título:** Right-side sugar stimulation recruits ~8,300 neurons; traced to il3LN6 (GABAergic) being excitatory in Connectivity_783

> Hi Philip and team,
>
> Thank you for releasing the model — it has been a great resource. While
> auditing it, we found a few things you might find useful. All are
> reproducible with the scripts linked below.
>
> 1. **Left/right asymmetry of the sugar response.** Stimulating all LB3 GRNs
>    on the left (150 Hz, 1 s) activates 446 neurons. The same on the right
>    activates 8,323, including 3,330 Kenyon cells and most of the antennal
>    lobe. We traced this to a single neuron: il3LN6 (left,
>    720575940632403986; known_nt = gaba in Schlegel et al. 2024; 1,805 output
>    connections, 12,671 synapses) has `Excitatory = 1` in
>    `Connectivity_783.parquet`. Flipping only this neuron's sign brings the
>    right-side response to 377 neurons.
> 2. **Other sign disagreements with literature transmitters.** 1,208 neurons
>    with known_nt GABA/glutamate are excitatory in the file (278,327 output
>    synapses; mostly Dm3, C2, Dm9 and antennal-lobe lLN2P_b), and 1,399 known
>    cholinergic neurons are inhibitory (e.g., L4, L5, T4d, T5d, Mi1).
> 3. **Olfactory regime.** An odor driving six glomeruli activates 66% of
>    Kenyon cells and 82% of PNs of non-stimulated glomeruli. Removing
>    excitatory chemical LN→PN and PN→PN synapses (~31k) gives 3.4% of Kenyon
>    cells and 0% off-target PNs.
>
> Importantly, all the validated predictions we re-tested still hold after
> these corrections: sugar→MN9, Rattle, Usnea, Clavicle and Fudog; the bitter
> veto; the partial Ir94e veto; and JO-CE→aBN1 versus JO-F.
>
> Code and details: https://github.com/amthedev/cerebro-mosca (`experimentos/14_auditoria.py`).
> Draft write-up: https://github.com/amthedev/cerebro-mosca/blob/main/artigo/preprint.md
>
> Best regards,
> Allan Matheus Silva Santos

---

## 2. Anotações FlyWire — github.com/flyconnectome/flywire_annotations/issues

**O que diz, em português:** o FlyWire chama o tipo LB1e de "amargo", mas o
BANC, outro conectoma, diz que o LB1e responde a aminoácidos (receptor Ir94e).
Além disso, o FlyWire junta em "LB3 açúcar/água" quatro subtipos que o BANC
separa. A mensagem é uma pergunta, não uma afirmação de erro.

**Título:** LB1e annotated as "bitter" — BANC annotates it as Ir94e (amino acids); LB3 subtypes

> Hi FlyWire annotation team,
>
> While comparing FlyWire v783 with BANC v888 (matched through BANC's
> `fafb_match` / `fafb_cell_type`), we noticed two taste-neuron annotation
> differences that may be worth a check:
>
> - `LB1e` has `cell_sub_class = bitter` in Supplemental File 1, while BANC
>   annotates LB1e as "amino_acids, Ir94e". Shiu et al. 2024 treat Ir94e GRNs
>   as a separate class from bitter.
> - `LB3` (122 neurons, "sugar/water") corresponds in BANC to LB3a (water,
>   ppk28), LB3b/LB3c (sugar, Gr64f) and LB3d (heavy metal, Ir47a). A finer
>   split in FlyWire would help modelling studies that stimulate "sugar" GRNs.
>
> Thank you for the annotations — they made this comparison possible.
> Allan Matheus Silva Santos · https://github.com/amthedev/cerebro-mosca

---

## 3. BANC — github.com/jasper-tms/the-BANC-fly-connectome/issues

**O que diz, em português:** o arquivo de neurotransmissores do BANC tem uma
coluna de "consenso por tipo" que contradiz os próprios neurônios em 68 tipos,
por exemplo células de Kenyon previstas como acetilcolina com consenso
"serotonina". Parece um erro na montagem do arquivo. A mensagem também inclui
duas observações úteis para quem simula o BANC: as diferenças de sinais em
relação ao FlyWire e a contagem de sinapses, que depende da tabela, v2 ou v3.

**Título:** `cell_type_neurotransmitter_predicted` contradicts neuron-level predictions for 68 cell types (e.g. Kenyon cells → serotonin)

> Hi BANC team,
>
> Thank you for releasing BANC. While simulating its brain portion with the
> Shiu et al. (2024) LIF model, we noticed something in
> `compiled_data/banc_888/banc_888_neurotransmitter_prediction_v2.csv` that
> looks like a processing issue:
>
> **1. Per-type consensus vs neuron-level predictions.** For 68 of the 2,595
> cell types (≥ 5 neurons) whose own neurons agree ≥ 90% on one transmitter,
> `cell_type_neurotransmitter_predicted` names a different transmitter, often
> with `cell_type_neurotransmitter_score = 1.0`. These types contain 9,088
> neurons. Examples (neurons; share of neuron-level calls → consensus):
>
> | cell_type | neurons | neuron-level majority | consensus |
> |---|---|---|---|
> | KCg-m | 1,448 | acetylcholine 99.1% | serotonin (1.00) |
> | KCab | 1,330 | acetylcholine 99.5% | glutamate (0.89) |
> | Tm1 | 860 | acetylcholine 99.1% | glutamate (0.84) |
> | JO-E | 364 | acetylcholine 96.7% | glutamate (1.00) |
> | ORN_DA1 | 224 | acetylcholine 99.6% | histamine (1.00) |
> | LPLC2 | 177 | acetylcholine 98.9% | glutamate (1.00) |
>
> The full list is produced by `experimentos/17_nt_banc.py` in https://github.com/amthedev/cerebro-mosca. It looks like the consensus may be joined to the wrong type in
> some cases, but you will know better.
>
> **2. For people simulating BANC (observations, not errors).** Compared with
> FlyWire v783 on 8,190 matched cell types, the predicted fast-transmitter
> sign differs for 7.0%: 292 types are inhibitory in FlyWire but excitatory in
> BANC, versus 115 the other way round. Also, 64 of 1,148 Johnston's-organ
> neurons are predicted GABAergic. Synapses per neuron on matched types are
> ~2.1× lower than in the FlyWire model file with the v2 edge list and ~1.7×
> lower with v3, and the ratio varies by region (2.5× on MN9 inputs vs 1.5× on
> giant-fiber inputs with v2). A single global scale is therefore not enough
> for LIF simulations: BANC becomes unstable while FlyWire does not.
>
> Details and code: https://github.com/amthedev/cerebro-mosca (`src/mosca/banc.py`,
> `experimentos/13_duas_moscas.py`, `experimentos/17_nt_banc.py`).
>
> Best,
> Allan Matheus Silva Santos
