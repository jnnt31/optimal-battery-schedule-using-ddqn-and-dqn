# BESS Energy Management Models

This repository contains DQN, Double DQN, dynamic-programming, and no-storage baseline implementations for battery energy storage dispatch.

## Data privacy

The original seasonal input spreadsheets are not included. Supply your own Excel files at runtime. Published result workbooks, CSV history files, and model checkpoints are included where available.

Input workbooks must contain these columns:

- `TIME (MINUTE)`
- `LOAD(MW)`
- `BUYING PRICE (USD/MWhr)`
- `SELLING PRICE (USD/MWhr)`
- One or more day columns named like `DAY 1 POWER (MW)`

## Install

```text
pip install -r requirements.txt
```

Python 3.10 or newer is recommended.

## Run a model

Pass one or more of your own Excel files and choose a separate output directory:

```text
python "DDQN/doubleDQN (1).py" --input-files "C:/data/july.xlsx" "C:/data/december.xlsx" --output-dir results/my-ddqn
python "dqn_v20/doubleDQN (1).py" --input-files "C:/data/july.xlsx" --output-dir results/my-dqn
python "DP/dp_bess_matched_to_ddqn.py" --input-files "C:/data/july.xlsx" --output-dir results/my-dp
python "no storage operation/no_storage_baseline_standalone.py" --input-files "C:/data/july.xlsx" --output-dir results/my-baseline
```

The scripts write generated outputs only to the directory passed with `--output-dir`.

## Repository contents

- `DDQN/`: Double DQN implementation and published outputs.
- `dqn_v20/`: DQN implementations and published outputs.
- `DP/`: dynamic-programming benchmark.
- `no storage operation/`: no-storage baseline.
- Root `*.xlsx` files: published comparison results.

Do not commit private datasets, credentials, local paths, IDE settings, caches, TensorBoard logs, or new generated run folders.
