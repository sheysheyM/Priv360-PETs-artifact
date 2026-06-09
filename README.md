# Priv360 Artifact

Artifact for:

**Priv360: Application-Oriented QoE-Optimized Client-Side Protection for 360-Viewer Identification**  
Sheyda Mirzakhani and Niklas Carlsson  
*To appear in Proceedings on Privacy Enhancing Technologies (PoPETs), 2026.*

This repository is prepared for the **PoPETs Artifact Available** badge. It provides the paper-related source code and documentation in a public, readable form.

This submission is **not** intended to claim full result reproduction. The raw datasets, generated paper outputs, and complete reproduction environment are not included in this repository.

Priv360 protects VR users from re-identification by injecting noise into the head movement data transmitted during 360° video streaming. A lightweight prediction step reconstructs a usable signal from the noise, maintaining streaming quality without exposing the true head pose. This artifact provides the source code for evaluating the privacy–utility tradeoff across multiple attacker models and noise levels.

## What is included

The artifact contains the code path corresponding to the paper's attack-and-defense evaluation:

1. **Clean telemetry / no defense** — attacker models are evaluated on the configured head-pose features without a privacy mechanism.
2. **Basic noisy telemetry defense** — noise is applied before attacker evaluation. This is a simple defense baseline.
3. **Prediction-assisted Priv360 defense** — noise is followed by the prediction/filtering path. This is the main Priv360 defense condition represented in this artifact.

The included attacker models are:

- **LSTM**
- **Transformer**
- **Random Forest**

## Repository tree

```text
.
├── README.md
├── RUN_AND_OUTPUT.md
├── ARTIFACT-APPENDIX.md
├── LICENSE
├── requirements.txt
└── priv360/
    ├── evaluate_attack_models_on_clean_noisy_prediction.py
    ├── apply_noise_and_prediction_defenses.py
    └── dataset_settings_and_loader.py
```

## Paper-to-code map

| Paper/artifact concept | Repository file |
|---|---|
| Clean attack baseline / no defense | `priv360/evaluate_attack_models_on_clean_noisy_prediction.py` |
| LSTM, Transformer, and Random Forest attackers | `priv360/evaluate_attack_models_on_clean_noisy_prediction.py` |
| Basic noisy telemetry defense baseline | `priv360/apply_noise_and_prediction_defenses.py` |
| Prediction-assisted Priv360 defense | `priv360/apply_noise_and_prediction_defenses.py` |
| Dataset settings and loading | `priv360/dataset_settings_and_loader.py` |
| Run commands and output format | `RUN_AND_OUTPUT.md` |

## Optional run example

The raw datasets are **not included**. These commands are provided for users who already have the expected dataset folder locally.

```bash
pip install -r requirements.txt

# attack without defense
python priv360/evaluate_attack_models_on_clean_noisy_prediction.py --dataset ds1 --model clean_lstm --data-root /path/to/dataset

# basic noisy telemetry defense baseline and prediction-assisted Priv360 defense
# (noisy_lstm and defense_lstm are aliases for the same sigma-sweep runner;
#  both produce acc_noisy_s* and acc_pred_s* columns in the same output CSV)
python priv360/evaluate_attack_models_on_clean_noisy_prediction.py --dataset ds1 --model noisy_lstm --data-root /path/to/dataset
```

More run commands and the CSV output format are described in [`RUN_AND_OUTPUT.md`](RUN_AND_OUTPUT.md).



## Citation & Acknowledgments

If you find this artifact or the Priv360 paper useful in your research, please consider citing:

```bibtex
@article{mirzakhani2026priv360,
  title={Priv360: Application-Oriented QoE-Optimized Client-Side Protection for 360-Viewer Identification},
  author={Mirzakhani, Sheyda and Carlsson, Niklas},
  journal={Proceedings on Privacy Enhancing Technologies (PoPETs)},
  year={2026},
  note={To appear}
}
```
