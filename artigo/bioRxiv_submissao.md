# Campos do formulário do bioRxiv (copiar e colar)

Arquivo: `artigo/preprint.pdf` (texto + figuras num único PDF).

**Article type:** New Results
**Subject area:** Neuroscience
**License:** CC BY 4.0 (a mais aberta, qualquer um pode reutilizar citando você)

**Title:**
Auditing a whole-brain Drosophila model across hemispheres and individuals: a single mis-signed neuron, a non-sparse olfactory regime, and limited cross-connectome reproducibility

**Author:** Allan Matheus Silva Santos — Independent researcher — allandevjr@gmail.com (corresponding author)

**Keywords:** connectome; Drosophila; whole-brain model; leaky integrate-and-fire; FlyWire; BANC; neurotransmitter; reproducibility

**Competing interests:** The author declares no competing interests.
**Funding:** None.
**Data and code availability:** All code is available at https://github.com/amthedev/cerebro-mosca. All data are public (FlyWire v783, Shiu et al. 2024 model repository, BANC v888 release) and are downloaded by `scripts/baixar_dados.sh`.
**AI use:** Code, analyses and a first draft of the text were produced with the help of an AI assistant (Claude, Anthropic). The author reviewed the code and results and takes responsibility for the content.

**Abstract (versão curta, se o formulário pedir menos palavras):**
Connectome-constrained whole-brain models are increasingly used to generate testable predictions. We audited the leaky integrate-and-fire model of the adult Drosophila brain (Shiu et al., 2024) using an exact event-driven reimplementation that runs a 1-s trial in under 1 s on one laptop core. All experimentally validated predictions we re-tested survive biologically motivated corrections. However, odors drive 66% of Kenyon cells in the original model (real flies: ~5–10%), and stimulating the right sugar receptor neurons recruits 8,323 neurons instead of ~450. A single GABAergic antennal-lobe neuron (il3LN6) labeled excitatory in the model file causes this; correcting it alone restores 377. Running identical protocols on a second fly's brain (BANC), cell-type-level agreement between flies (Spearman ρ ≤ 0.32) is lower than between the two hemispheres of one fly (ρ = 0.54), and depends on synapse-table calibration and transmitter labels, which disagree for 7% of matched types. We also report a transmitter-consensus column in the BANC release that contradicts neuron-level predictions for 68 cell types. Whole-brain model predictions should be checked across hemispheres and connectomes.
