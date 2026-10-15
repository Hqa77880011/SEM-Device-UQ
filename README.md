# Uncertainty Propagation from SEM Boundaries to MOSFET Electrical Characteristics

Implementation of **Uncertainty Propagation from SEM Boundaries to MOSFET Electrical Characteristics**.

The boundary network predicts a mean SDF and smooth low-rank spatial basis. Gaussian anchor NLL uses the full low-rank covariance, and geometric samples remain spatially coherent. The execute path evaluates a normalized mixture-over-boundary compatibility likelihood by bounded Metropolis sampling, then propagates process parameters through a supplied process/device adapter. Electrical I-V metrics and interval scores are computed from propagated draws; a proprietary TCAD solver is never replaced by fabricated currents.

## Install

Use Python 3.10 or later. Create and activate a virtual environment, then install:

```bash
python -m pip install -e .
```

## Prepare data

`x[N,1,H,W]` contains registered SEM crops, binary `y`, and physical `sdf`. Freeze anchor coordinates, pixel pitch, and device registration using training data. For posterior execution provide boundary-anchor draws, discrepancy/surrogate covariance, four parameter bounds, and a Python adapter exporting geometry(theta) and device(theta). The adapter runs the characterized process and Sentaurus/device model and returns an I-V curve on the declared voltage grid.

Arrays use numeric NPZ format, loaded without pickle. `group[N]` identifies independent subjects, devices, scenes, lots, or source images. Related samples must share a group. Optional `split[N]` codes 0/1/2/3 explicitly assign training/validation/calibration/test (three-role input using 0/1/2 remains supported); otherwise ordered groups are allocated 60/15/10/15 to training/validation/calibration/test. Test data never select model parameters or checkpoints.

```bash
sem-uncertainty demo --output data/example.npz --count 96 --config config.json
sem-uncertainty prepare --input data/example.npz --output data/prepared
```

The demo defines a small synthetic problem with the documented tensor interface. Use application data in that schema for the paper's evaluation settings.

## Train and execute

```bash
sem-uncertainty train --input data/prepared --output runs/full --config config.json --variant full --epochs 100 --batch-size 16
sem-uncertainty evaluate --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/evaluation
sem-uncertainty plot --input runs --output runs/figures
```

Training uses AdamW and selects the minimum validation objective. `model.pt` stores parameters and configuration; `history.json` records training/validation objectives. Evaluation exports predictions and mechanism tensors to `predictions.npz` and computed scores to `metrics.json`. Plotting exports comparison charts from those actual evaluation files.

`--device cpu` is the default; `--device cuda` selects CUDA. `--lr`, `--epochs`, `--batch-size`, and `--seed` control optimization. Method parameters are in `config.json`; `--set key=value` overrides one value using JSON syntax. Use identical splits and seeds across comparisons.

## Calibration

For interval or classification confidence calibration, use independent groups excluded from fitting and checkpoint selection:

```bash
sem-uncertainty calibrate --input data/prepared/calibration.npz --model runs/full/model.pt --output runs/full/calibrated.pt
```

Evaluate `calibrated.pt` with the same evaluation command. Classification uses temperature scaling; numeric intervals use a 90% group-maximum split-calibration expansion. Insufficient independent units retain infinite intervals. Pretraining checkpoints can initialize supervised training with `train --init path/to/model.pt`.

## Baselines and ablations

Available variants: `full`, `baseline`, `no_spatial_basis`, `no_likelihood`. `baseline` removes the paper-specific mechanism while retaining the input and prediction task. Other variants disable their named component.

```bash
sem-uncertainty train --input data/prepared --output runs/baseline --config config.json --variant baseline
sem-uncertainty evaluate --input data/prepared/test.npz --model runs/baseline/model.pt --output runs/baseline/evaluation
```

## Metrics and implementation conventions

Classification reports accuracy, balanced accuracy, macro-F1, NLL, confidence ECE, and worst-group F1; binary tasks add AUROC and AUPRC. Unknown labels use -1 and enable unknown AUROC and OSCR. Segmentation reports Dice, IoU, precision, recall, MAE, Brier/ECE, two-pixel boundary F1, symmetric contour distance, and HD95. Reconstruction reports RMSE, MAE, signed bias, unit-range PSNR, and Gaussian-window SSIM. Each method adds its mechanism metrics where applicable.

Parameter counts and forward timing describe the executing process. Use batch size one and a declared device configuration for device latency comparisons. Physical energy and scores from external task evaluators require measured inputs; they are not inferred from parameter counts. All prediction arrays remain available for task-specific evaluation.

The implementation uses the explicit core operations described above with configurable compact networks. Numeric conventions and application-specific information are stated with the method and data schema. The parameter file contains the current defaults.

## Execute the mechanism

```bash
sem-uncertainty execute --input data/posterior_problem.json --output runs/execution --variant full
```

The example is an explicit analytical/simulated input specification, not an experiment result. Replace it with the schema described above for application data. The execute path writes the actual selected actions, constraints, and mechanism metrics.
