> **Arquivado.** Para usar de novo: mova `src/mosca/*.py` e `experimentos/*.py` daqui de volta para o projeto,
> mova `vendor/flygym_repo` para `vendor/` e rode `uv add flygym`.

# cerebro: mosca-da-fruta virtual

O cérebro inteiro de uma mosca (*Drosophila*), copiado neurônio por neurônio do
conectoma FlyWire (138.639 neurônios, 15 milhões de conexões), controlando um
corpo físico simulado (NeuroMechFly v2 no MuJoCo) que vê pelos olhos compostos,
cheira pelas antenas, come, aprende, dorme e foge de predadores num mundo com
dia e noite.

## Como rodar

```bash
uv run python experimentos/01_validar_cerebro.py   # compara nosso motor com o modelo original (Brian2)
uv run python experimentos/02_mapear_saidas.py     # que neurônios de saída respondem a cada sentido
uv run python experimentos/03_vida.py              # Vida 1.0 -> resultados/vida.mp4
uv run python experimentos/05_aprender.py          # condicionamento olfativo só no cérebro
uv run python experimentos/06_vida_aprende.py      # Vida 2.0 -> resultados/vida_aprende.mp4
uv run python experimentos/07_avaliar_memoria.py --grupo treinada --n 6   # estatística
uv run python experimentos/07_avaliar_memoria.py --grupo ingenua --n 6
uv run python experimentos/08_grafico_memoria.py   # -> resultados/memoria_trajetorias.png
uv run python experimentos/09_um_dia.py            # Vida 3.0: um dia inteiro -> resultados/um_dia.mp4
uv run python experimentos/10_medula.py            # descendentes -> pernas pela medula real (BANC)
```

Para ver ao vivo numa janela 3D (MuJoCo), com o estado do cérebro na tela:

```bash
uv run mjpython experimentos/ao_vivo.py                     # aprende e depois é testada
uv run mjpython experimentos/ao_vivo.py --cenario simples   # Vida 1.0
uv run mjpython experimentos/ao_vivo.py --cenario dia       # Vida 3.0: um dia (a cena escurece à noite)
```

No macOS o `mjpython` precisa da biblioteca do Python dentro do `.venv`
(`vendor/flygym_repo/scripts/link_libpython_dylib_macos.sh`).

Dados da medula (BANC, ~385 MB) em `dados/banc/`, baixados de
`gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/`
(`banc_888_meta.feather`, `banc_888_edgelist_simple_v2.feather`,
`banc_888_neurotransmitter_prediction_v2.csv`).

## Estrutura

| arquivo | o que faz |
|---|---|
| `src/mosca/cerebro.py` | motor do cérebro: modelo LIF de Shiu et al. 2024, reescrito em numba, passo a passo |
| `src/mosca/neuronios.py` | catálogo de neurônios por tipo (anotações FlyWire) |
| `src/mosca/biologia.py` | correções biológicas ao modelo original (moduladores, sinais, lobo antenal) |
| `src/mosca/memoria.py` | corpo cogumelo: memória de cheiros ensinada pela dopamina |
| `src/mosca/corpo.py` | corpo NeuroMechFly, probóscide articulada, salto de fuga, arena, comida, predador |
| `src/mosca/visao.py` | olhos compostos (721 omatídeos por olho) -> detector de aproximação -> LPLC2 |
| `src/mosca/locomocao.py` | gerador de passos (CPG) do FlyGym, reescrito 5x mais rápido; frequência dos passos sobe com o comando |
| `src/mosca/voo.py` | voo: asas batendo a 200 Hz, cinemática inversa, aerodinâmica de asa de inseto (Robofly), controle de voo |
| `src/mosca/ponte.py` | sentidos -> neurônios sensoriais; neurônios descendentes -> comandos do corpo |
| `src/mosca/estados.py` | relógio (dia/noite) e sono (pressão de sono) |
| `src/mosca/vida.py` | ciclo de vida: cérebro + corpo + metabolismo + memória + sono, em sessões |
| `src/mosca/medula.py` | medula real (conectoma BANC, 28 mil neurônios) acoplada ao cérebro, com sinapse elétrica Fibra Gigante -> TTMn |
| `src/mosca/mundos.py` | mundo "um dia": arena, comidas, zona de perigo, predadores, relógio |
| `src/mosca/video.py` | vídeo com painel de sentidos, neurônios, energia, memória e sono |

## O que vem do conectoma e o que é atalho

| canal | origem |
|---|---|
| açúcar -> neurônios gustativos -> **MN9** -> probóscide estica | conectoma |
| saciedade: com a barriga cheia o MN9 para sozinho | conectoma + fome modulando o ganho do açúcar |
| cheiro -> neurônios olfativos de cada antena -> lobo antenal -> PNs -> células de Kenyon | conectoma (sem a excitação lateral química, ver abaixo) |
| memória: Kenyon + dopamina -> sinapses KC->MBON enfraquecem | conectoma + regra de plasticidade |
| olhos compostos -> expansão de uma sombra no céu -> **LPLC2** -> **Fibra Gigante** | retina real + detector simplificado + conectoma |
| Fibra Gigante (cérebro) -> Fibra Gigante (medula BANC) -> sinapse elétrica -> **TTMn** -> salto | conectoma + sinapse elétrica da literatura |
| Fibra Gigante -> sinapse elétrica -> **PSI** -> **motoneurônios de voo DLM** -> decolagem | conectoma + sinapses identificadas da literatura |
| voo: asas batendo no ar, sustentação e arrasto de cada asa | física (elemento de pá com coeficientes medidos do Robofly) |
| controle do voo (altitude, velocidade, direção, pouso) e halteres | atalho (no lugar dos circuitos de voo) |
| ameaça de um lado -> **DNa01/DNa02 do lado oposto** -> vira para longe | conectoma |
| sono: pressão (S) + relógio circadiano (C); pressão -> ER5, dormindo -> dFB | modelo de dois processos; conectoma recebe; parar de andar é atalho |
| açúcar/ameaça -> dopamina (PAM/PPL1) | atalho (testado: açúcar -> PAM 0 Hz; amargo -> PPL1 ~2 Hz no modelo) |
| lembrança do cheiro -> aproximar/evitar (klinotaxia) | atalho: leitura dos MBONs (testado: pelo conectoma a memória não chega separável aos descendentes) |
| virar para o cheiro de comida (comparando as antenas) | atalho (testado: o conectoma leva o lado do cheiro aos DNa, mas só os DNa esquerdos respondem) |
| andar mais quando está com fome | atalho |
| movimento das pernas (CPG, não os motoneurônios reais); frequência dos passos sobe com o comando (6 a 30 mm/s) | atalho |
| estabilização no ar (halteres) e reflexo de se endireitar | atalho |

## Resultados

- Motor do cérebro validado contra o original: correlação 0,999 entre as taxas
  de disparo, MN9 a 77,7 Hz contra 78,8 Hz no Brian2. Roda 1 s de cérebro em
  ~2,8 s num Apple M4.
- Vida 1.0 (`03_vida.py`): com fome, segue o cheiro, come até ficar satisfeita
  e foge do predador.
- Vida 2.0 (`06`/`07`): aprende que um cheiro significa comida e outro perigo;
  no teste, 6 de 6 moscas treinadas vão ao cheiro bom e evitam o ruim (índice
  +0,99), as ingênuas andam ao acaso (-0,55, n=6).
- Visão: o predador só aparece nos últimos ~80 ms (a 40 mm ele ocupa menos que
  um omatídeo). A retina detecta, a Fibra Gigante dispara (~100-140 Hz) e a
  mosca salta, cai em pé e escapa. O predador tem ~120 ms de atraso de reação:
  quem não salta a tempo é pega.
- Vida 3.0 (`09_um_dia.py`): um dia comprimido em 210 s com amanhecer,
  anoitecer e noite; a comida acaba e brota em outro lugar; predadores atacam
  mais na zona de perigo; a mosca dorme à noite e acorda de manhã.

## Problemas encontrados no modelo original (e correções em `biologia.py`)

1. **Moduladores como excitação rápida.** Dopamina, serotonina e octopamina
   (~2 milhões de sinapses) estão como excitatórias rápidas. Um único neurônio
   serotonérgico (CSD) liga o lobo antenal inteiro com qualquer cheiro.
   Correção: 616 neurônios moduladores (por identidade) saem da transmissão rápida.
2. **Sinais trocados.** ~3 mil neurônios GABA/glutamatérgicos estão como
   excitatórios. Correção: sinal pelo neurotransmissor da literatura (known_nt).
3. **Cuidado com a previsão automática de neurotransmissor:** ela diz que as
   5.172 células de Kenyon são dopaminérgicas (são colinérgicas). Usar essa
   previsão cegamente desliga o corpo cogumelo inteiro. (Na medula BANC, ela
   diz que a maioria dos motoneurônios é GABAérgica; eles são glutamatérgicos.)
4. **Lobo antenal inunda.** Qualquer cheiro ativava ~90% dos neurônios de
   projeção de *todos* os glomérulos. Causa: ~31 mil sinapses químicas
   excitatórias laterais (neurônio local colinérgico -> PN e PN -> PN). Na mosca
   essa excitação lateral é sobretudo por junções elétricas fracas. Sem elas
   (e com toda a inibição local mantida): 80-93% dos PNs do glomérulo certo
   ativos contra 1% dos outros; células de Kenyon 1,5-4% ativas, 0-1% de
   sobreposição entre cheiros, como na mosca real.
5. **Achado:** no conectoma, o cheiro de comida *inibe* o MN9 (com cheiro
   presente é preciso mais açúcar para a probóscide estender).

## Memória (`memoria.py`)

- Fiação toda do conectoma: ORN -> lobo antenal -> PN -> Kenyon -> MBON,
  dopamina -> MBON (os compartimentos saem sozinhos dos dados:
  PAM01 -> MBON01 γ5, PPL101 -> MBON11 γ1pedc...).
- Regra: Kenyon ativa + dopamina no compartimento -> sinapse KC->MBON enfraquece.
- Condicionamento só no cérebro (`05_aprender.py`, lobo antenal real):
  cheiro+açúcar -> +1,88 (atrai), cheiro+punição -> -0,61 (repele), controles ~0.
- Como a mosca usa a memória para se mover: klinotaxia (se a lembrança do
  cheiro melhora, segue reto; se piora, vira) e busca local onde a lembrança é
  boa. Só usa a saída dos MBONs, sem saber de onde vem cada cheiro.

## Medula real (`medula.py`, `10_medula.py`)

- Conectoma BANC (outro indivíduo, fêmea): 28 mil neurônios da medula e 872
  mil conexões, com os descendentes casados por tipo com os do FlyWire. Os 391
  motoneurônios de perna vêm anotados com perna e função (ex.: "flexionar
  fêmur-tíbia"), que correspondem às articulações do corpo.
- O sinal passa: DNa01/DNa02 de um lado ativam motoneurônios das pernas do
  mesmo lado (esquerdo: 5-7 Hz contra 0-1 Hz no direito); MDN ativa extensores
  coxa-trocânter nas pernas da frente. Mas é fraco (poucos Hz): os
  motoneurônios recebem mais inibição do que excitação, como na mosca, e na
  caminhada real eles são liberados em ritmo por circuitos (CPGs) e pela
  realimentação das pernas, que o conectoma estático sozinho não gera.
- Limite do conectoma: Fibra Gigante -> motoneurônio de salto (TTMn) não
  aparece, porque essa ligação é uma sinapse *elétrica* (junção comunicante),
  invisível no conectoma químico. Ela foi acrescentada como camada elétrica
  (King & Wyman 1980; Allen et al. 2006): cada disparo da Fibra Gigante faz o
  TTMn do mesmo lado disparar e a mosca saltar.
- A medula roda acoplada ao cérebro durante a vida: os descendentes do BANC
  disparam na taxa dos seus equivalentes do FlyWire (879 tipos casados).
- Achado: as duas Fibras Gigantes do BANC são muito conectadas entre si (~620
  sinapses); sem cansaço sináptico, no modelo LIF elas se disparam para sempre.
  Com depressão na saída da Fibra Gigante (habituação, como na mosca) o
  circuito responde uma vez e se apaga.

## Voo (`voo.py`)

- Asas com 3 articulações cada; a batida é descrita como na biologia (ângulo
  de varredura no plano horizontal + ângulo de ataque em torno da
  envergadura, que vira na inversão) e convertida nas articulações por
  cinemática inversa numérica (as articulações são encadeadas e mudam de eixo
  quando a asa abre).
- "Músculo" da asa rígido (ressonância ~1 kHz): com o servo mole do começo a
  asa entrava em ressonância na própria frequência da batida.
- Aerodinâmica: o modelo de fluido do MuJoCo (quase-estacionário simples) dá
  só ~0,4x o peso, porque asas de inseto dependem do vórtice no bordo de
  ataque. Usamos elemento de pá com os coeficientes medidos na asa-robô
  Robofly (Dickinson, Lehmann & Sane 1999): a mosca gera 1,5-4x o peso.
- Controle (atalho): altitude pelo ângulo de ataque, velocidade inclinando o
  corpo (como um drone, com correção integral), direção pela diferença de
  ângulo de ataque entre as asas, halteres estabilizando, pouso freando e
  descendo devagar, desvio da parede da arena.
- Na fuga: um disparo da Fibra Gigante -> TTMn (salto) + PSI -> 5
  motoneurônios de voo DLM do mesmo lado -> decolagem de 0,5 s para longe da
  ameaça. Voar gasta ~5x mais energia que andar.

## Memória -> comportamento pelo conectoma (teste)

MBONs de "aproximar" ativam 64 neurônios descendentes (incluindo DNa01/DNa02
e DNb05); os de "evitar" (glutamatérgicos) quase nenhum. Mas cheirar um odor
já ativa esses descendentes por outras vias (corno lateral) e, depois do
condicionamento, a mudança neles fica em ±1 Hz (ruído). Por isso a leitura
"lembrança -> aproximar/evitar" continua sendo um atalho declarado.

## Próximos passos

1. Medula com realimentação das pernas (proprioceptores) para tentar gerar o
   ritmo da caminhada no próprio conectoma, no lugar do CPG.
2. Controle do voo pelos descendentes de voo e motoneurônios de direção
   das asas (BANC tem 24 de direção e 12 de tensão), no lugar do controlador.
3. Detector de movimento da retina pelos circuitos T4/T5 (modelo flyvis).

## Créditos

Conectoma FlyWire (Dorkenwald et al. 2024, Schlegel et al. 2024), modelo do
cérebro (Shiu et al. 2024, MIT), conectoma BANC (Bates, Phelps, Kim, Yang et
al. 2026), NeuroMechFly v2 / FlyGym (Wang-Chen et al. 2024, Apache 2.0), MuJoCo.
