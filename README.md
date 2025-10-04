F3-Histo: Feature Flow-aware Temporal Evolution Dynamics Supervision for Enhanced Pathological Image Classification
This repository contains the official implementation of the F3-Histo framework, as presented in the paper "Feature Flow-aware Temporal Evolution Dynamics Supervision for Enhanced Pathological Image Classification" published in Medical Image Analysis. The framework introduces a novel approach to histopathology image classification by modeling feature flow-aware temporal evolution dynamics, leveraging diffusion models, timestep-modulated mamba (TMM), dynamic graph feature aggregation (DGFA), and adaptive class balance (ACB).

![F3-Histo Framework](./kuangjia.pdf)


Abstract
Histopathology images are critical for disease diagnosis, but traditional pixel-based models often fail to capture the complex spatiotemporal relationships inherent in pathological state transitions. The F3-Histo framework addresses this limitation by introducing the first unified feature flow-aware temporally evolving dynamic supervision model for histopathology image classification. By integrating diffusion models to model temporal evolution, TMM for long-range dependency capture, DGFA for adaptive feature aggregation, and ACB for robust class balance, F3-Histo achieves enhanced classification performance, offering a novel paradigm for medical image analysis.
Introduction
Pathological image classification is pivotal for accurate disease diagnosis, yet existing methods, primarily pixel-based, struggle to model the dynamic evolution of pathological states over time. These methods often rely on static feature extraction, neglecting the temporal dynamics and complex spatial relationships within histopathology images. The F3-Histo framework overcomes these challenges through a unified approach that captures feature flow-aware temporal evolution dynamics. Key innovations include:

Diffusion-based Temporal Modeling: Utilizes diffusion models to simulate the temporal progression of pathological features, enabling robust feature extraction across timesteps.
Timestep-modulated Mamba (TMM): Captures long-range dependencies in feature sequences, modulated by temporal embeddings to enhance dynamic modeling (Section 3.3).
Dynamic Graph Feature Aggregator (DGFA): Employs graph neural networks to adaptively aggregate multi-scale features, preserving spatial and temporal relationships (Section 3.4).
Adaptive Class Balance (ACB): Introduces a dynamic sampling strategy based on Fréchet Inception Distance (FID) to address class imbalance and improve classification robustness (Section 3.5).

F3-Histo represents a significant advancement in histopathology image analysis, offering a scalable and effective solution for clinical diagnostics.
Installation

Clone the repository:git clone https://github.com/HBnothave/Feature-Flow-Aware.git
cd Feature-Flow-Aware


Install dependencies:pip install -r requirements.txt


Place datasets in the data/ directory (see below for details).

Requirements
See requirements.txt for a full list of dependencies. Key requirements include:

Python 3.9
PyTorch 2.0.1
torch-geometric
NumPy, torchvision, scikit-learn, scipy

Dataset Setup
Place histopathology datasets in the data/ directory with subfolders for each dataset (e.g., data/BreakHis/, data/NCT-CRC-HE-100K/). The code supports multiple datasets, organized by class subdirectories. Ensure datasets are downloaded and structured as expected by utils/data_loader.py.
Training
To train the F3-Histo model, run:
python train.py --dataset BreakHis --data_dir data/BreakHis --batch_size 16 --epochs 200 --lr 1e-4


Arguments:
--dataset: Dataset name (e.g., BreakHis, NCT-CRC-HE-100K, GasHisSDB-160, ROSE).
--data_dir: Path to the dataset directory.
--batch_size: Batch size (default: 16).
--epochs: Number of training epochs (default: 200).
--lr: Learning rate (default: 1e-4).



Model checkpoints will be saved in the checkpoints/ directory based on the best validation F1 score.
Evaluation
To evaluate a trained model, run:
python train.py --dataset BreakHis --data_dir data/BreakHis --evaluate --checkpoint checkpoints/best_model.pth

This outputs classification metrics (accuracy, precision, recall, F1 score).
