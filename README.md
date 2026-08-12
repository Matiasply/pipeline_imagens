# Sobre
Este projeto foi feito por Dennynson Scheydt, Erick Batista e Matias Araújo como proposta para o projeto final da disciplina Introdução ao Processamento Digital de Imagens. Ele tem como finalidade criar uma pipeline de pré-processamento para leitura labial.

# Dependências

cv2, mediapipe, numpy, pytorch e lipnet (submodulo)

# Execução

Foi utilizado o dataset GRID Corpus, nessa estrutura:

```
dataset/
├── lombardgrid_alignment/
│   └── lombardgrid/
│       └── alignment/
├── lombardgrid_audio/
│   └── lombardgrid/
│       └── audio/
├── lombardgrid_front/
│   └── lombardgrid/
│       └── front/
└── lombardgrid_json/
    └── lombardgrid/
        └── json/
```

O pipeline irá processar os vídeos e armazená-los em output_dir, após processar todos vídeos irá analisar o vídeo original e o vídeo processado com o modelo LipNet, armazenando as avaliações em output_dir/evaluation. O comando para executá-lo tem o seguinte formato: (--limit 0 para executar todos vídeos)

```bash
.venv/bin/python src/pipeline.py   \
--output-dir results/<pasta-de-output>   \
--face-model face_landmarker.task   \
--model-path third_party/LipNet-PyTorch/pretrain/LipNet_unseen_loss_0.44562849402427673_wer_0.1332580699113564_cer_0.06796452465503355.pt   \
--limit <quantidade-de-exemplos>
```

O script report_evaluation.py faz um resumo das avaliações realizadas, para executá-lo faça:

```bash
.venv/bin/python src/report_evaluation.py \
  --input results/<pasta-de-output>/evaluation/sentence_evaluation_summary.json \
  --output results/<pasta-de-output>/evaluation/effectiveness_report.json
```
