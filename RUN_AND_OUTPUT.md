# Run Commands and Output Format

These commands are optional. They document how the artifact code is used by someone who already has the expected datasets locally.

This repository is submitted for the **Artifact Available** badge. It does not include the raw datasets or a full reproduction environment.

## 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Set the dataset path

Point `--data-root` to the folder expected by `priv360/dataset_settings_and_loader.py`.

```bash
DATA_ROOT=/path/to/dataset
```

Use `--dataset ds1` or `--dataset ds2`, depending on which dataset folder is available locally.

## 3. Attack without defense

These commands evaluate the attackers on clean telemetry.

```bash
python priv360/evaluate_attack_models_on_clean_noisy_prediction.py --dataset ds1 --model clean_lstm --data-root "$DATA_ROOT" --output ds1_clean_lstm.csv
python priv360/evaluate_attack_models_on_clean_noisy_prediction.py --dataset ds1 --model clean_transformer --data-root "$DATA_ROOT" --output ds1_clean_transformer.csv
python priv360/evaluate_attack_models_on_clean_noisy_prediction.py --dataset ds1 --model clean_rf --data-root "$DATA_ROOT" --output ds1_clean_rf.csv
```

## 4. Basic noisy telemetry defense baseline

These commands evaluate the attack models after applying the basic noisy telemetry defense. The telemetry is perturbed before attacker evaluation, without using the prediction/filtering step as the main Priv360 defense.

> **Note:** `noisy_lstm` and `defense_lstm` (and the corresponding `_transformer` and `_rf` variants) route to the same sigma-sweep runner. Both model names are accepted as aliases for the same experiment, which outputs both the basic noisy baseline (`acc_noisy_s*`) and the prediction-assisted Priv360 defense (`acc_pred_s*`) columns in a single CSV. This allows direct comparison of the two conditions under the same sigma sweep.

```bash
python priv360/evaluate_attack_models_on_clean_noisy_prediction.py --dataset ds1 --model noisy_lstm --data-root "$DATA_ROOT" --output ds1_noisy_lstm.csv
python priv360/evaluate_attack_models_on_clean_noisy_prediction.py --dataset ds1 --model noisy_transformer --data-root "$DATA_ROOT" --output ds1_noisy_transformer.csv
python priv360/evaluate_attack_models_on_clean_noisy_prediction.py --dataset ds1 --model noisy_rf --data-root "$DATA_ROOT" --output ds1_noisy_rf.csv
```

## 5. Prediction-assisted Priv360 defense

These commands are equivalent to section 4 above (`noisy_*` and `defense_*` are aliases for the same runner). Use whichever name is clearer in your workflow. The output CSV contains both the basic noisy baseline and the prediction-assisted Priv360 defense columns together.

```bash
python priv360/evaluate_attack_models_on_clean_noisy_prediction.py --dataset ds1 --model defense_lstm --data-root "$DATA_ROOT" --output ds1_prediction_defense_lstm.csv
python priv360/evaluate_attack_models_on_clean_noisy_prediction.py --dataset ds1 --model defense_transformer --data-root "$DATA_ROOT" --output ds1_prediction_defense_transformer.csv
python priv360/evaluate_attack_models_on_clean_noisy_prediction.py --dataset ds1 --model defense_rf --data-root "$DATA_ROOT" --output ds1_prediction_defense_rf.csv
```

## 6. Output format

The program writes a CSV file when `--output` is provided. If no output path is provided, the default filename is:

```text
<dataset>_<model>_results.csv
```

All output CSVs include the following metadata columns in addition to the result columns described below:

| Column | Meaning |
|---|---|
| `fold` | Fold number (1-indexed). |
| `test_video` | Video/session held out for testing in that fold. |
| `n_cls` | Number of user classes in this fold. |
| `n_train` | Number of training rows in this fold. |
| `n_test` | Number of test rows in this fold. |
| `time_min` | Runtime for the fold in minutes. |

### Clean attack output

For `clean_lstm`, `clean_transformer`, and `clean_rf`, the result column is:

| Column | Meaning |
|---|---|
| `acc_clean` | Attack accuracy on clean telemetry, without a defense. |

### Noisy and prediction-assisted output

For `noisy_lstm`, `noisy_transformer`, `noisy_rf`, `defense_lstm`, `defense_transformer`, and `defense_rf`, result columns are written for each sigma value in the sweep (`SIGMAS_NOISY = [1.0, 5.0, 20.0]`):

| Column | Meaning |
|---|---|
| `acc_noisy_s<SIGMA>` | Attack accuracy after the basic noisy telemetry defense at noise level `<SIGMA>`. |
| `acc_pred_s<SIGMA>` | Attack accuracy after the prediction-assisted Priv360 defense at noise level `<SIGMA>`. |

For example, with the default sigma sweep the output includes columns:

```text
acc_noisy_s1.0, acc_pred_s1.0
acc_noisy_s5.0, acc_pred_s5.0
acc_noisy_s20.0, acc_pred_s20.0
```

The noisy and prediction-assisted columns are written in the same CSV so the basic noisy defense baseline and the main Priv360 defense can be compared under the same sigma sweep.
