# Dados

Este diretório não é versionado. Os dados de entrada, modelos de velocidade,
checkpoints e figuras são gerados localmente.

## Estrutura esperada

```text
data/
├── velocity_model/
├── observed_data/
├── checkpoints/
└── images/
```

## Modelo BP

O script `scripts/velocity_models/gerando_BP.py` baixa
`vel_z6.25m_x12.5m_exact.segy.gz` e gera:

```text
data/velocity_model/BP_model_freq_3_v1.pth
data/velocity_model/BP-smooth-linear_freq_3_v1.pth
```

Execute:

```bash
python scripts/velocity_models/gerando_BP.py
```

O fluxo BP usa fonte Ricker de 3 Hz (`--freq 3.0`).

## Modelo Marmousi

O script `scripts/velocity_models/gerando_marmousi.py` baixa `vp`, `vs` e
`density` do Marmousi II e gera:

```text
data/velocity_model/marmousi2-model-true-large_freq_5.pth
data/velocity_model/marmousi2-model-smooth-linear_freq_5.pth
```

Execute:

```bash
python scripts/velocity_models/gerando_marmousi.py
```

## Dados observados

Os dados observados são gerados pelos próprios scripts de inversão quando
`--run_all 1` é usado. Exemplo:

```bash
python scripts/inversion/run_imaging.py --vel_model bp --freq 3.0 --run_all 1
```

## Observações

- O fluxo de 6 camadas não faz parte deste repositório.
- Os arquivos de velocidade BP e Marmousi são derivados dos dados originais e
  não devem ser commitados.
