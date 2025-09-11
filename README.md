# WiFi Flow Matching (Wi-Flow-Model)

A deep learning project implementing Conditional Flow Matching for WiFi-based indoor floor plan prediction. This project uses flow matching techniques to generate indoor layouts based on WiFi heatmaps and trajectories.

## Project Structure

### Core Training and Evaluation Files

- **`train.py`** - Main training script for the flow matching model.
- **`train_ab_hm.py`** - Ablation study training script focusing on heatmap-only generation (without trajectory information).
- **`train_ab_traj.py`** - Ablation study training script focusing on trajectory-only generation (without heatmap information).

### Validation Scripts

- **`val.py`** - Main validation script for evaluating model performance.
- **`val_ab_hm.py`** - Validation script for heatmap-only ablation studies.
- **`val_ab_traj.py`** - Validation script for trajectory-only ablation studies.
- **`test.py`** - Testing script for evaluating trained models on real-world test datasets.
- **`eval.py`** - Evaluation script that computes metrics like SSIM (Structural Similarity Index) and IoU (Intersection over Union) for comparing generated and ground truth images.

### Utility and Data Handling

- **`dataloader.py`** - Contains the Dataloader class for loading WiFi heatmaps, trajectories, and target images. Handles data preprocessing and augmentation.
- **`utils.py`** - Utility functions including exponential moving average (EMA) for model parameter updates.

### Flow Matching Framework (`torchcfm/`)

- **`conditional_flow_matching.py`** - Core implementation of various conditional flow matching algorithms.
- **`optimal_transport.py`** - Optimal transport utilities for flow matching.
- **`utils.py`** - Additional utilities for the flow matching framework.
- **`version.py`** - Version information for the torchcfm package.
- **`models/`** - Directory containing model architectures:
  - **`unet/`** - UNet model implementations used as the backbone for flow matching.

### Datasets

#### Scale 100 Datasets
- **`Dataset_Scale100_SEPE/`**
  - **`train/`** - Training data with Condition_1 Heatmap, Condition_2 Trajectory, and Target floor plan.
  - **`test/`** - Test data with the same structure
  - **`Selected_50_train/`** - Subset of 50 selected training samples
  - **`Selected_50_test/`** - Subset of 50 selected test samples
  - **`id2idx.txt`** - Mapping file for sample IDs to indices

- **`Dataset_Scale100_SExPE/`**
  - Same structure as SEPE dataset but for different experimental conditions

#### Real-World Datasets (`Dataset_rw/`)
- Each experiment contains:
  - **`Condition_1/`** - WiFi Heatmap
  - **`Condition_2/`** - Robot Trajectory

### Results and Outputs

#### Training Results (`results/`)
Contains training outputs organized by timestamp and experiment type.

## Usage

1. **Training**: Use `train.py` for main experiments, or `train_ab_*.py` for ablation studies
2. **Validation**: Validate the weight by `val.py` scripts
3. **Evaluation**: Use `eval.py` to compute metrics on test results
4. **Testing**: Run `test.py` with appropriate checkpoint and real-world dataset

The project focuses on WiFi-based indoor positioning using advanced generative modeling techniques, specifically conditional flow matching, to predict the indoor floor plan.

## References

This project builds upon the conditional flow matching implementation from:
- **Conditional Flow Matching Repository**: [https://github.com/atong01/conditional-flow-matching](https://github.com/atong01/conditional-flow-matching)

**Paper References:**
- A. Tong, N. Malkin, G. Huguet, Y. Zhang, J. Rector-Brooks, K. Fatras, G. Wolf, Y. Bengio. "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport," 2023. [arXiv:2302.00482](https://arxiv.org/abs/2302.00482)
- Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, M. Le. "Flow Matching for Generative Modeling," ICLR 2023. [Paper](https://openreview.net/forum?id=PqvMRDCJT9t)

