F3-Histo: Feature Flow-aware Temporal Evolution Dynamics Supervision for Enhanced Pathological Image Classification
This repository contains the official implementation of the paper "Feature Flow-aware Temporal Evolution Dynamics Supervision for Enhanced Pathological Image Classification" published in Medical Image Analysis. The F3-Histo framework introduces a novel feature flow-aware dynamic modeling approach for histopathology image classification, leveraging diffusion models, timestep-modulated mamba (TMM), dynamic graph feature aggregation (DGFA), and adaptive class balance (ACB) to achieve state-of-the-art performance.

Paper Overview
F3-Histo is the first unified feature flow-aware temporally evolving dynamic framework for histopathology image classification. It addresses limitations of pixel-based models by capturing temporal dynamics of pathological state transitions, modeling complex spatiotemporal relationships, and achieving robust performance across diverse datasets.

Key Contributions:
Pioneers a feature flow-aware temporally evolving dynamic supervision framework.
Introduces an adaptive dynamic graph feature aggregation strategy using graph neural networks.
Demonstrates state-of-the-art performance on multiple histopathological datasets.



For detailed methodology, results, and discussions, refer to the paper (replace with actual DOI).
Datasets
The F3-Histo framework was evaluated on four histopathological datasets:

BreakHis: 5,534 breast cancer images, 8 classes, multi-magnification (40×, 100×, 200×, 400×).
NCT-CRC-HE-100K: 100,000 colorectal cancer images, 9 tissue classes.
GasHisSDB-160: 33,284 gastric cancer images, binary classification.
ROSE: 4,240 pancreatic cancer images, binary classification, private dataset.


Download the public datasets from their respective sources:

BreakHis: Link
NCT-CRC-HE-100K: Link
GasHisSDB-160: Link

Place the datasets in the data/ directory with subfolders BreakHis/, NCT-CRC-HE-100K/, GasHisSDB-160/, and ROSE/ (for private dataset users).
Installation

Clone the repository:git clone https://github.com/HBnothave/Feature-Flow-Aware.git
cd Feature-Flow-Aware


Install dependencies:pip install -r requirements.txt


Ensure the datasets are placed in the data/ directory.

Requirements
See requirements.txt for a full list of dependencies. Key requirements include:

Python 3.9
PyTorch 2.0.1
CUDA 12.1 (for GPU acceleration)
NumPy, torchvision, scikit-learn, etc.

Training
To train the F3-Histo model, run the following command:
python train.py --dataset BreakHis --data_dir data/BreakHis --batch_size 16 --epochs 200 --lr 1e-4


Arguments:
--dataset: Choose from BreakHis, NCT-CRC-HE-100K, GasHisSDB-160, or ROSE.
--data_dir: Path to the dataset directory.
--batch_size: Batch size (default: 16).
--epochs: Number of training epochs (default: 200).
--lr: Learning rate (default: 1e-4).



Model checkpoints will be saved in the checkpoints/ directory based on the best validation F1 score.
Evaluation
The trained model can be evaluated using:
python train.py --dataset BreakHis --data_dir data/BreakHis --evaluate --checkpoint checkpoints/best_model.pth

This will output accuracy, precision, recall, and F1 score metrics.
Results
F3-Histo achieves state-of-the-art performance across all datasets:

BreakHis (MD): Accuracy up to 98.10% (40×), F1 up to 97.95%.
BreakHis (MI): Accuracy 95.48%, F1 95.64%.
NCT-CRC-HE-100K: Accuracy 99.41%, F1 99.37%.
GasHisSDB-160: Accuracy 99.30%, F1 99.16%.
ROSE: Accuracy 95.86%, F1 95.80%.


Visualizations

Class Activation Maps (CAM): Visualize regions of interest across diffusion timesteps.
t-SNE Visualizations: Show feature separability across datasets.

Citation
If you use this code or the F3-Histo framework, please cite:
@article{zhang2025f3histo,
  title={Feature Flow-aware Temporal Evolution Dynamics Supervision for Enhanced Pathological Image Classification},
  author={Zhang, Peng and Zhao, Heyang and Liu, Zeyu and Ma, Chenbin and Xue, Qianqian and Liu, Xingyu and Wu, Tian and Zhang, Guanglei and Wang, Wenjian},
  journal={Medical Image Analysis},
  year={2025},
  doi={XXXX}
}

License
This project is licensed under the MIT License.
Contact
For questions, please contact:

Guanglei Zhang: guangleizhang@buaa.edu.cn
Wenjian Wang: wjwang@sxu.edu.cn
