# Como publicar (passo a passo, você faz)

Nada foi publicado nem enviado. Tudo depende de você.

## 0. Antes de tudo: revisar
- Leia o `preprint.md` (ou o `preprint.pdf`) inteiro. Seu nome já está lá;
  falta trocar `[e-mail]` (o bioRxiv exige um e-mail de contato). Depois gere
  o PDF de novo:
  `uv run --with markdown python scripts/gerar_pdf.py`.
- O artigo do BANC está citado como preprint do bioRxiv
  (doi:10.1101/2025.07.31.667571). Se ele já saiu numa revista, troque pela
  versão publicada.
- Se puder, peça a um neurocientista (professor, pós-graduando) que leia antes.
  Um coautor da área dá muito mais credibilidade e pega erros que eu posso ter
  deixado passar.
- Rode de novo, na sua máquina, `experimentos/13` a `16` e confira que os
  números batem.

## 1. Código no GitHub (feito)
https://github.com/amthedev/cerebro-mosca. Para enviar mudanças novas:
`git add -A && git commit -m "o que mudou" && git push`. Os dados (~1 GB) não
vão para o GitHub (`.gitignore`); quem clonar roda `scripts/baixar_dados.sh`.

## 2. DOI para o código (Zenodo, grátis)
Em zenodo.org, entre com a conta do GitHub, ative o repositório e crie um
"release" no GitHub. O Zenodo gera um DOI citável.

## 3. Preprint no bioRxiv (grátis)
- Converta para PDF (ex.: cole no Google Docs com as figuras de
  `resultados/figuras/` e exporte; ou use pandoc).
- biorxiv.org → Submit → categoria "Neuroscience". Pesquisador independente
  pode enviar. Eles fazem uma triagem de alguns dias antes de publicar.
- Declare o uso de IA (já está na seção "AI assistance").

## 4. Avisar as equipes (feito)
As três issues estão abertas (links no README). Acompanhe as respostas pelo
GitHub (você recebe notificação por e-mail) e responda com educação.

## 5. Divulgar
Um fio curto (Bluesky/X/LinkedIn) com a figura 2B: "um neurônio com sinal
errado faz o modelo de cérebro inteiro da mosca explodir". Marque os autores
com respeito, sem sensacionalismo.
