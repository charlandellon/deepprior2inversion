# Deep Bayesian inference for seismic inversion

Implementação do fluxo de inversão Bayesiana profunda para imageamento sísmico,
com suporte aos modelos BP, Marmousi e 4 camadas.

## Estrutura

```text
deepprior2inversion/
├── dp4inversion/          # pacote principal
├── scripts/
│   ├── inversion/         # treinamento/inversão
│   ├── modeling/          # simulação e plotagem de dados sísmicos
│   ├── velocity_models/   # geração dos modelos de velocidade
│   ├── analysis/          # índices estatísticos e RMSE
│   ├── visualization/     # amostras, vídeos e paleta
│   └── launchers/         # comandos prontos por modelo
├── data/                  # dados e resultados (não versionado)
├── environment.yml
├── setup.py
├── run.sh                 # pipeline ponta a ponta
└── README.md
```

## Instalação

```bash
git clone https://github.com/Uncertainty-Quantification-Project/deepprior2inversion.git
cd deepprior2inversion
conda env create -f environment.yml
conda activate deepprior2inversion
pip install -e .
```

## Dados

Os dados não são versionados. O diretório `data/` é criado automaticamente
pelo pacote e deve conter:

```text
data/
├── velocity_model/
├── observed_data/
├── checkpoints/
└── images/
```

Para baixar e gerar os modelos de velocidade:

```bash
python scripts/velocity_models/gerando_BP.py
python scripts/velocity_models/gerando_marmousi.py
```

Consulte `data/README.md` para os arquivos esperados e URLs de origem.

## Uso rápido

### BP

```bash
bash scripts/launchers/bp/run_plot_seismic_BP.sh
bash scripts/launchers/bp/run_BP.sh
bash scripts/launchers/bp/run_statistical_index_BP.sh
```

O exemplo BP usa fonte Ricker de 3 Hz, aquisição de 10 s e SNR de 2,4 dB na
figura de sísmica limpa/ruidosa.

### Marmousi

```bash
bash scripts/launchers/marmousi/run_marmousi.sh
bash scripts/launchers/marmousi/run_marmousi_convencional.sh
bash scripts/launchers/marmousi/run_plot_seismic_marmousi.sh
```

### 4 camadas

```bash
bash scripts/launchers/4layers/run_4layers.sh
bash scripts/launchers/4layers/run_plot_seismic_4layers.sh
```

## Pipeline completo (BP ou Marmousi)

O script `run.sh` executa o fluxo ponta a ponta:

1. baixa e gera o modelo de velocidade;
2. gera os dados sísmicos;
3. executa a inversão;
4. calcula os índices estatísticos.

```bash
bash run.sh bp
bash run.sh marmousi
```

Exemplo de execução rápida:

```bash
MAX_ITR=100 RUN_STATS=0 bash run.sh bp
```

Variáveis úteis: `MAX_ITR`, `NSRC`, `NREC`, `SIGMA`, `FREQ`, `CUDA`,
`RUN_ALL`, `GENERATE_MODEL`, `RUN_STATS`, `PYTHON_BIN`.

## Opções úteis

- `--tmax`: tempo máximo de aquisição, em segundos.
- `--snr`: SNR alvo da wavelet ruidosa, em dB.
- `--vel_model`: `bp`, `marmousi` ou `4layers`.
- `--freq`: frequência dominante da fonte Ricker, em Hz.

## Saídas

As figuras e checkpoints são gravados em `data/images/` e
`data/checkpoints/`, organizados pelo nome do experimento.

## Licença

MIT. Consulte o arquivo `LICENSE`.
