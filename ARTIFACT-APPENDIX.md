# Artifact Appendix

## Paper

**Priv360: Application-Oriented QoE-Optimized Client-Side Protection for 360-Viewer Identification**  
Sheyda Mirzakhani and Niklas Carlsson  
*To appear in Proceedings on Privacy Enhancing Technologies (PoPETs), 2026.*

## Requested badge

We request the **Artifact Available** badge.

We do **not** request the **Artifact Functional** or **Artifact Reproduced** badges for this submission.

## Artifact availability

This artifact is intended to be hosted as a public repository at a stable location, such as GitHub, GitLab, Zenodo, or another accepted artifact hosting service.

The repository includes:

- source code relevant to the Priv360 paper;
- documentation explaining the paper-to-code mapping;
- a license file;
- this artifact appendix.

## Artifact scope

This artifact provides the paper-aligned source code and documentation for:

- clean telemetry attack evaluation, without a defense;
- basic noisy telemetry defense baseline;
- prediction-assisted Priv360 defense;
- LSTM, Transformer, and Random Forest attacker models.

The artifact is intentionally focused on availability and reuse. It does not include raw datasets, generated outputs, plotting scripts, Docker files, or unrelated QoE-only scripts.

## Main files

- `README.md`: repository entry point, paper-to-code map, and badge scope.
- `RUN_AND_OUTPUT.md`: optional run commands and CSV output format.
- `priv360/evaluate_attack_models_on_clean_noisy_prediction.py`: main attack/defense evaluation runner.
- `priv360/apply_noise_and_prediction_defenses.py`: noise and prediction-assisted defense code.
- `priv360/dataset_settings_and_loader.py`: dataset settings and loading helpers.

## Available badge checklist

- Public artifact link: to be provided through the final public repository or archival record.
- License: included as `LICENSE`.
- Relevance to paper: the artifact contains the attack models and defense scenarios corresponding to the Priv360 paper.
- Artifact appendix: included as `ARTIFACT-APPENDIX.md`.

## Notes on functionality and reproduction

This submission is for artifact availability. The artifact does not claim that reviewers can reproduce all numerical results from the paper using only this repository.

Full numerical reproduction requires access to the datasets used in the paper and the full experimental environment. The included commands are provided to document how the code is organized and how it can be run by users who have the expected datasets locally.

## Citation

```bibtex
@article{mirzakhani2026priv360,
  title={Priv360: Application-Oriented QoE-Optimized Client-Side Protection for 360-Viewer Identification},
  author={Mirzakhani, Sheyda and Carlsson, Niklas},
  journal={Proceedings on Privacy Enhancing Technologies (PoPETs)},
  year={2026},
  note={To appear}
}
```
