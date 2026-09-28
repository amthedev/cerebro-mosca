# Auditing a whole-brain *Drosophila* model across hemispheres and individuals: a single mis-signed neuron, a non-sparse olfactory regime, and limited cross-connectome reproducibility

**Allan Matheus Silva Santos**¹

¹ Independent researcher · [e-mail]

*Preprint draft — September 2026. Not peer reviewed.*

---

## Abstract

Connectome-constrained whole-brain models are increasingly used to generate
testable predictions. The leaky integrate-and-fire (LIF) model of the adult
*Drosophila* brain by Shiu et al. (2024), built on the FlyWire connectome,
predicted sensorimotor pathways for feeding and grooming that were largely
confirmed experimentally. Here we ask how robust such predictions are to
(i) errors in neurotransmitter labels, (ii) the single free synaptic weight
parameter, and (iii) the individual animal whose connectome is used. We
reimplemented the model in an exact event-driven form that reproduces the
original spikes while running a 1-s trial in under 1 s on one laptop CPU core
(the original code takes ~16 min per trial, including network construction).
All experimentally validated predictions we tested (sugar → MN9 and four named
interneurons, bitter veto, Ir94e partial veto, JO-CE → aBN1) survive a set of
biologically motivated corrections. However, the audit revealed two hidden
problems. First, odors drive 66% of Kenyon cells and 82% of projection neurons
of non-stimulated glomeruli, far from the sparse, glomerulus-specific coding of
real flies; removing ~31,000 excitatory chemical lateral synapses in the
antennal lobe restores 3.4% Kenyon-cell activity and 0% off-target projection
neurons. Second, stimulating the *right* sugar receptor neurons (the original
study used the left) recruits 8,323 neurons instead of ~450. A single cell, the
left il3LN6 — a GABAergic antennal-lobe local neuron with 12,671 output
synapses that is labeled excitatory in the model's connectivity file — accounts
for this: correcting its sign alone reduces the response to 377 neurons. In the
same file, 1,208 neurons with literature-established GABA/glutamate identity are
excitatory and 1,399 established cholinergic neurons are inhibitory. Finally,
we ran identical protocols on the brain portion of a second female connectome
(BANC). Comparing firing rates of 8,209 matched cell types, repeat simulations
agree almost perfectly (Spearman ρ = 0.98), the two hemispheres of the same fly
agree moderately (ρ = 0.54), and the two flies agree weakly (ρ ≤ 0.32).
BANC's released edge lists carry fewer synapses per neuron on matched types
(2.1× fewer with the v2 table used in the BANC paper, 1.7× with the newer v3
table), with a regionally variable ratio. Its neurotransmitter predictions
also disagree in sign with FlyWire for 7% of matched types, leaving it less
inhibited. A global weight rescaling drives BANC into runaway activity,
whereas FlyWire is stable across 0.7–1.5× the published weight. Per-type
calibration with type-level transmitter labels restores the looming → giant
fiber and sugar → MN9 pathways (best: v3 table with BANC's own per-type
majority labels, MN9 at 46 vs 98.5 Hz), but not JO → aBN1. Cell-type-level
agreement between the flies never exceeds ρ = 0.32 in any of seven versions. We also found that the
per-type transmitter consensus distributed with BANC contradicts the
neuron-level predictions of 68 highly consistent cell types (9,088 neurons,
including Kenyon cells, LPLC2 and Johnston's-organ neurons). We conclude that
whole-brain LIF predictions should be checked across hemispheres and
connectomes, and that neurotransmitter labels are a dominant source of
fragility. Code and all analyses are openly available.

---

## Introduction

The complete synaptic wiring diagram of an adult female *Drosophila* brain
(FlyWire; Dorkenwald et al., 2024; Schlegel et al., 2024) enabled a
brain-wide LIF model in which every neuron shares the same parameters and
synaptic weights are proportional to synapse counts, with a sign given by the
predicted neurotransmitter (Shiu et al., 2024; Eckstein et al., 2024). Despite
its simplicity, the model predicted neurons involved in taste-evoked feeding
and antennal grooming, and 91% of 164 tested predictions were consistent with
experiments (Shiu et al., 2024).

Such models are now being ported to new connectomes, including the
brain-and-nerve-cord connectome BANC (Bates et al.) and the male CNS, and
reused in embodied simulations. Several open-source projects have already run
the same model on multiple connectomes and noted that synapse counts differ
between datasets (e.g., the *brainlab* and *flymsg* repositories). What is
missing is a systematic answer to three questions: which model predictions
depend on uncertain neurotransmitter labels; how close the model operates to
unstable regimes; and whether its cell-type-level predictions replicate in a
second individual, compared against a proper baseline of within-animal
variability.

We address these questions with (1) an exact, fast reimplementation;
(2) an audit of the published model against both its own validated
predictions and textbook facts it was not tested on; (3) a sweep of the
synaptic weight; and (4) a cross-individual comparison in which the two
hemispheres of each fly serve as a within-animal control.

## Results

### 1. An exact event-driven engine makes the model interactive on a laptop

We reimplemented the model's equations with exact linear integration, the same
update order as Brian2, the same 1.8-ms delay and 2.2-ms refractory period,
and the Brian2 "unless refractory" semantics, under which inputs arriving
during refractoriness are discarded (omitting this inflates rates by ~30%).
Firing rates from the sugar experiment matched the original Brian2 model with
r = 0.999 (MN9: 77.7 vs 78.8 Hz). An event-driven mode integrates only neurons
away from rest and draws random numbers in the same order as the dense mode;
across sugar, looming and odor protocols it produced 0 differing spikes. An
automatic mode switches between dense and event-driven integration depending on
the fraction of active neurons. For the published sugar trial (150 Hz, 1 s, one
Apple M4 core), the original code took 967 s, most of it spent building the
network, which the original code does for every trial. Our engine took 0.48 s
(event-driven) or 1.59 s (dense), plus 0.4 s to load the connectome
(**Fig. 1**). With dense activity (odor, looming), it runs at 1.6–2.1 s per
simulated second.

### 2. Validated predictions are robust, but olfaction runs in a non-biological regime

We applied four corrections, each individually and all together:
(a) dopaminergic, serotonergic and octopaminergic neurons (616 identified by
cell class or literature transmitter) were removed from fast transmission, since
they act via metabotropic receptors; (b) signs were set from
literature-established transmitters (`known_nt`) where available, otherwise
from high-confidence predictions (Kenyon cells forced cholinergic; Barnstedt et
al., 2016); (c) histamine was made inhibitory (Hardie, 1989); (d) excitatory
chemical synapses from antennal-lobe local neurons and projection neurons onto
projection neurons (~31,000 connections) were removed, since lateral excitation
between glomeruli is mediated mainly by weak electrical coupling (Yaksi &
Wilson, 2010; Huang et al., 2010).

All experimentally validated predictions we tested held under every correction
(**Table 1**). By contrast, the original model failed two well-established
properties of olfactory coding that it was not designed to capture. An odor
activating six glomeruli recruited 66% of Kenyon cells (real flies: ~5–10%;
Honegger et al., 2011) and 82% of projection neurons from non-stimulated
glomeruli. Only correction (d) fixed both (3.3% Kenyon cells; 1% off-target
projection neurons; with all corrections, 3.4% and 0%) (**Fig. 2A**). Across
the whole brain, the corrections left responses to sugar, JO and looming nearly
unchanged (r = 0.99 with the original), while the odor response shrank from
8,492 to 1,298 active neurons (r = 0.78).

**Table 1. Audit of the original model (FlyWire v783, 150 Hz, 1 s).**

| test | original | all corrections |
|---|---|---|
| MN9 firing with sugar (Hz) | 69.0 | 63.0 |
| Rattle / Usnea / Clavicle / Fudog with sugar (Hz) | 48 / 77 / 33 / 6 | 48 / 77 / 31 / 6 |
| MN9, sugar + bitter, relative to sugar alone | 0.01 | 0.01 |
| MN9, sugar + Ir94e, relative to sugar alone | 0.17 | 0.14 |
| aBN1 with JO-CE / JO-F (Hz) | 70 / 0 | 70 / 0 |
| Giant fiber with LPLC2 (Hz) | 122.5 | 116.0 |
| Kenyon cells active with odor (%) | **66.4** | 3.4 |
| PNs active, stimulated / other glomeruli (%) | 100 / **82** | 100 / 0 |
| Active neurons, left / right sugar GRNs | 446 / **8,323** | 453 / 370 |

### 3. One mis-signed neuron makes the right-side sugar response explode

The original study stimulated the left sugar receptor neurons. Stimulating the
same cell type (LB3) on the right recruited 8,323 active neurons versus 446 on
the left, including 3,330 Kenyon cells and most of the antennal lobe
(**Fig. 2B**). Removing modulators did not help (7,743), whereas correcting
signs did (370). Tracing which sign-corrected neurons fired first, then
correcting them one at a time, identified a single cause: the left il3LN6
(FlyWire 720575940632403986), an antennal-lobe local neuron annotated as
GABAergic, with 1,805 output connections and 12,671 output synapses, which is
excitatory in the model's connectivity file. Making only this neuron inhibitory
reduced the right-side response from 8,323 to 377 neurons.

More broadly, the signs in the model's connectivity file disagree with
literature-established transmitters for 2,607 neurons: 1,208 GABA/glutamate
neurons are excitatory (278,327 output synapses; mostly optic-lobe Dm3, C2 and
Dm9, and antennal-lobe lLN2P_b and il3LN6) and 1,399 cholinergic neurons are
inhibitory (e.g., L4, L5, T4d, T5d, Mi1).

### 4. FlyWire is stable across synaptic weights; BANC is not

Scaling the single weight parameter (0.275 mV per synapse) by 0.7–1.5× changed
FlyWire's sugar response smoothly (337–588 active neurons; Spearman ρ with the
published weight ≥ 0.85), and looming similarly (400–1,065 neurons; ρ ≥ 0.78).
The antenna protocol was the most sensitive (ρ = 0.40 at 1.5×). The BANC brain,
by contrast, showed abrupt transitions to brain-wide activity: at 1.4× for
looming (271 → 2,770 neurons) and between 1.6× and 1.8× for sugar (82 → 4,229)
(**Fig. 3**).

### 5. Two flies agree less than two hemispheres of the same fly

We extracted the brain-equivalent portion of BANC v888: 151,000 neurons, with
8.9 M connections and 24.7 M synapses in the v2 edge list used by the BANC paper
(synapses of size ≥ 5 voxels) or 10.6 M connections and 31.1 M synapses in the
newer v3 edge list (updated detector, size ≥ 10), versus 138,639 neurons,
15.1 M connections and 54.5 M synapses in the FlyWire model file. Each BANC
neuron was given a FlyWire cell type through its matched FlyWire neuron
(64,278 neurons) or a consistently translated type name (64,145), yielding
8,209 shared types. The same corrections were applied. Six protocols
stimulated the same FlyWire cell type in both flies (sugar LB3, bitter LB1a–d,
food odor, CO₂, LPLC2 and JO-C/E), on each side. For each (cell type, side),
we computed the mean firing rate, excluded the stimulated types, and compared
conditions by the Spearman correlation over types active (> 2 Hz) in either
condition and by the Jaccard index of active sets. Three baselines bracket the
answer: the same fly with another random seed (noise ceiling), the same fly
with left versus mirrored right stimulation (within-animal variability), and
FlyWire versus BANC (between animals).

Repeat simulations agreed at ρ = 0.98. Hemispheres agreed at ρ = 0.54 in
FlyWire and 0.47 in BANC. Between flies, agreement was at best ρ = 0.32
(Jaccard 0.28; **Fig. 4A**). BANC, used as is, reproduced looming → giant fiber
at reduced strength (50 Hz with v2, 77.5 Hz with v3, vs 116 Hz in FlyWire),
but sugar evoked no MN9 activity and JO-C/E evoked no aBN1 activity with
either table (**Fig. 4B**).

Two data differences explained most of the gap in activity levels. First, the
BANC edge lists carry fewer synapses per neuron on matched types: a median
ratio of 2.12 with v2 (interquartile range 1.43–3.08) and 1.72 with v3. The
ratio is regional: with v2 it is 2.51× on MN9 inputs versus 1.52× on
giant-fiber inputs. The partners and signs of the MN9 inputs are conserved;
only their counts differ. Since v2 and v3 differ by 26% in total synapses, this
difference reflects synapse detection and size thresholds, not necessarily
biology. Second, neurotransmitter predictions disagree in sign for 7.0% of the
8,190 matched types with outputs. Of these, 292 types are inhibitory in
FlyWire but excitatory in BANC, versus 115 the other way round, so the mean
inhibitory fraction of input is lower in BANC (0.39 vs 0.44). For example, 64
of 1,148 BANC Johnston's-organ neurons, which are mechanosensory and expected
to be cholinergic, are predicted GABAergic.

Because of these two differences, simple fixes fail. A global 2.12× weight
rescaling drove BANC into runaway activity (5,000–8,000 active neurons for five
of six stimuli, versus 71–1,606 in FlyWire; ρ = 0.03). So did per-type
calibration, which scales each neuron's inputs to the FlyWire mean of its type
(runaway for odor, looming and antenna stimuli; ρ = 0.09). Per-type
calibration combined with type-level transmitter labels worked better. We
tested two sets of labels: FlyWire's per-type signs, and the majority
transmitter of each BANC cell type computed from BANC's own neuron-level
predictions. With FlyWire signs (v2), activity returned to FlyWire-like levels
(e.g., 406 vs 453 active neurons for sugar), and sugar → MN9 (21.5 Hz) and
looming → giant fiber (97 Hz) were restored. BANC's own type-majority labels
gave the strongest MN9 response with the v3 table (46 Hz; giant fiber 135 Hz),
without any FlyWire information except for the calibration. Still, no version
exceeded ρ = 0.32 between flies; the versions that restored MN9 reached
ρ = 0.15–0.30. JO-C/E → aBN1 stayed near silent in all of them (≤ 4 Hz, vs
34 Hz in FlyWire).

**A transmitter-consensus inconsistency in the BANC release.** The BANC
per-neuron prediction file also provides a per-cell-type consensus
(`cell_type_neurotransmitter_predicted`). In 68 of the 2,595 types whose own
neurons agree ≥ 90% on one transmitter, this consensus names a different
transmitter, often with a confidence of 1.0. These 68 types contain 9,088
neurons: for example, KCg-m (1,448 neurons, 99.1% predicted cholinergic;
consensus serotonin), KCab (consensus glutamate), Tm1, LPLC2, ORN_DA1 and the
JO-E and JO-F Johnston's-organ types. We therefore computed type majorities
ourselves. Under standard sign rules, using the consensus column as a shortcut for neuron
signs would silence or invert major sensory and memory pathways.

## Discussion

The validated predictions of the FlyWire LIF model are robust to the
corrections we tested, which supports the original conclusions. The audit
nevertheless shows that the model's behavior can hinge on individual
neurotransmitter labels. One of 138,639 neurons, labeled with the wrong sign,
switches a sensory response from local to brain-wide. The olfactory system
operates far from its known sparse regime unless lateral excitatory chemical
synapses are removed. Any use of the model beyond the published circuits
(olfactory learning, embodied control, other sensory modalities) should
therefore start from corrected signs and include left–right controls.

The cross-individual comparison places a ceiling on how much cell-type-level
detail such a model can be trusted to predict. Even within one animal,
mirrored stimulation yields only moderately similar cell-type responses
(ρ ≈ 0.5). This reflects genuine left–right differences in wiring and
reconstruction, amplified by threshold dynamics. Across animals, agreement is
lower still and depends strongly on how two connectomes are calibrated against
each other. Robust predictions, such as looming → giant fiber, replicate across
hemispheres and animals, and these are the ones to prioritize for experiments.
Because neurotransmitter labels differ between datasets for the same cell types,
and because even a released consensus can be internally inconsistent,
cross-dataset consensus labels (and experimental verification for high-impact
neurons, such as il3LN6) would benefit all connectome-based models. Synapse
counts are likewise not directly comparable across datasets, or even across
synapse tables of one dataset, and cross-dataset simulations need an explicit
calibration step.

**Limitations.** All neurons share one set of LIF parameters, and there are no
gap junctions and no neuromodulation. Type
matching between datasets is imperfect, and per-type calibration uses FlyWire as
the reference. Seven BANC versions were tried; the conclusions do not depend
on picking the best one. Results depend on the > 2 Hz activity threshold and on the
chosen protocols. BANC's brain portion omits ascending input from the nerve
cord, as FlyWire does. None of the new claims has been tested *in vivo*.

## Methods

**Data.** FlyWire v783 connectivity and completeness files from the original
model repository (Shiu et al., 2024; MIT license); neuron annotations from
Schlegel et al. (2024), Supplemental File 1. BANC v888 metadata, edge lists
(`banc_888_meta.feather`, `banc_888_edgelist_simple_v2.feather`,
`banc_888_edgelist_simple_v3.feather`) and neurotransmitter predictions
(`banc_888_neurotransmitter_prediction_v2.csv`) from the BANC public release
(Harvard Dataverse doi:10.7910/DVN/7WTH1N). BANC neurons were included if their region was not the ventral
nerve cord or if they were ascending neurons (the FlyWire volume contains their
brain axons). Autapses were removed (absent in the FlyWire model file). Sides
missing in BANC for ~13,000 neurons (mostly optic lobe) were inferred from soma
position far from the midline.

**Model.** LIF with the parameters of Shiu et al.: v_rest = v_reset = −52 mV,
threshold −45 mV, τ_m = 20 ms, τ_syn = 5 ms, refractory period 2.2 ms,
delay 1.8 ms, 0.275 mV per synapse, dt = 0.1 ms, and Poisson input with a
250× multiplier. Signs follow the original rule (GABA and glutamate inhibitory,
all else excitatory) unless corrected as described. Trials lasted 1 s from rest.

**Corrections, protocols and metrics** are implemented in `src/mosca/biologia.py`,
`src/mosca/banc.py` and `experimentos/13–15` and `17`. BANC type-majority
labels: for each BANC cell type, the most frequent neuron-level prediction,
with the fraction of agreeing neurons as its confidence (the same confidence
thresholds as for FlyWire: 0.5 for fast transmitters, 0.8 for modulators). Named neurons (Rattle, Usnea,
Clavicle, Fudog, aBN1) use the root IDs from the original repository, all
unchanged in v783.

**Code availability.** https://github.com/amthedev/cerebro-mosca. `scripts/baixar_dados.sh` downloads all
public data. `experimentos/12`–`17` regenerate every number and figure.

**AI assistance.** Code, analyses and a first draft of this text were produced
with the help of an AI assistant (Claude, Anthropic). The author reviewed the
code and results and takes responsibility for the content.

## Figures

![Figure 1](../resultados/figuras/fig1_motor.png)

**Figure 1.** Wall-clock time for one 1-s trial of the published sugar
experiment on one Apple M4 core. The original code rebuilds the network for
every trial.

![Figure 2](../resultados/figuras/fig2_auditoria.png)

**Figure 2.** Audit of the original model. (A) Kenyon cells active with an odor
under each correction; the shaded band is the range reported for real flies.
(B) Active neurons when stimulating left versus right sugar GRNs (log scale).
Correcting the sign of a single neuron (left il3LN6) removes the asymmetry.

![Figure 3](../resultados/figuras/fig3_escala.png)

**Figure 3.** Active neurons as the weight per synapse is scaled, in FlyWire and
in BANC (v2), for three stimuli.

![Figure 4](../resultados/figuras/fig4_duas_moscas.png)

**Figure 4.** FlyWire versus BANC. (A) Cell-type-level agreement (mean over six
stimuli) for repeat simulations, mirrored hemispheres of FlyWire, and the two
flies, for seven BANC versions. (B) Key outputs in each BANC version; black
bars are FlyWire.

## References

- Barnstedt O, et al. (2016) Memory-relevant mushroom body output synapses are cholinergic. *Neuron* 89:1237–1247.
- Bates AS, Phelps JS, Kim M, Yang HH, et al. (2025) Distributed control circuits across a brain-and-cord connectome. *bioRxiv* doi:10.1101/2025.07.31.667571.
- Dorkenwald S, et al. (2024) Neuronal wiring diagram of an adult brain. *Nature* 634:124–138.
- Eckstein N, et al. (2024) Neurotransmitter classification from electron microscopy images at synaptic sites in *Drosophila melanogaster*. *Cell* 187:2574–2594.
- Hampel S, et al. (2015) A neural command circuit for grooming movement control. *eLife* 4:e08758.
- Hardie RC (1989) A histamine-activated chloride channel involved in neurotransmission at a photoreceptor synapse. *Nature* 339:704–706.
- Honegger KS, Campbell RAA, Turner GC (2011) Cellular-resolution population imaging reveals robust sparse coding in the *Drosophila* mushroom body. *J Neurosci* 31:11772–11785.
- Huang J, et al. (2010) Functional connectivity and selective odor responses of excitatory local interneurons in *Drosophila* antennal lobe. *Neuron* 67:1021–1033.
- Schlegel P, et al. (2024) Whole-brain annotation and multi-connectome cell typing of *Drosophila*. *Nature* 634:139–152.
- Shiu PK, et al. (2024) A *Drosophila* computational brain model reveals sensorimotor processing. *Nature* 634:210–219.
- Stimberg M, Brette R, Goodman DFM (2019) Brian 2, an intuitive and efficient neural simulator. *eLife* 8:e47314.
- Yaksi E, Wilson RI (2010) Electrical coupling between olfactory glomeruli. *Neuron* 67:1034–1047.
- Software: github.com/Pronexsteam/brainlab; github.com/gianlucamazza/flymsg (accessed September 2026).
