# SuperImagem Colab

Ampliação de imagens em **2×, 4× e 8×** usando uma GPU NVIDIA T4 no Google Colab.

## Recursos

- RealESRGAN x4plus oficial para fotos e uso geral.
- RealESRGAN x4plus Anime 6B para ilustrações e estampas.
- Processamento CUDA FP16 em blocos de 400 px com sobreposição.
- PNG transparente preservado sem converter o fundo para branco.
- Comparação entre original e resultado usando os arquivos completos.
- Visualização ajustada à tela ou em 100% dos pixels reais.
- Progresso visual por blocos.

## Executar no Colab

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/diegogomespi/superimagem-colab/blob/main/SuperImagem_GPU_T4_Colab.ipynb)

1. Abra o notebook pelo botão acima.
2. Selecione **GPU T4** no tipo de ambiente de execução.
3. Execute as células na ordem.
4. A interface será aberta dentro do notebook.

O primeiro processamento baixa o peso oficial do modelo escolhido. Arquivos enviados e resultados ficam somente no armazenamento temporário da sessão do Colab.

Mais detalhes: [colab/LEIA-ME-COLAB.md](colab/LEIA-ME-COLAB.md).
