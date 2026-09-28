#!/usr/bin/env bash
# Baixa os dados públicos usados pelo projeto (~1 GB).
#   - modelo e conectoma FlyWire v783 do artigo de Shiu et al. 2024 (GitHub, MIT)
#   - anotações FlyWire de Schlegel et al. 2024
#   - conectoma BANC v888 (Bates, Phelps, Kim, Yang et al.)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p vendor dados/banc

if [ ! -d vendor/Drosophila_brain_model ]; then
  git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model.git vendor/Drosophila_brain_model
fi

curl -L -o dados/flywire_neuron_annotations.tsv \
  https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv

B=https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888
for f in banc_888_meta.feather banc_888_edgelist_simple_v2.feather; do
  curl -L -o "dados/banc/$f" "$B/$f"
done
echo "pronto. Os caches (.npz) são criados na primeira execução."
