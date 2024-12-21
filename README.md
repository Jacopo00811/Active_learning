# Abalation study on active learning
## Overview
This project, for the course Introduction to Deep Learning (Fall 2024) at DTU, investigates the effectiveness of different active learning (AL) algorithms in reducing annotation costs while maintaining model performance. We focus on image classification using the CIFAR-10 dataset and ResNet-18 architecture.

## Abstract
Active Learning (AL) is a machine learning approach that focuses on selecting the most informative data points for labeling, aiming to achieve optimal model performance while minimizing the amount of labeled data required. This is particularly useful for reducing the high manual labor costs of annotating large, unlabeled datasets for deep learning models.

In this study, we compared different AL algorithms using the CIFAR-10 dataset and the ResNet-18 architecture, consisting of experiments using uncertainty-based, representative-based, and hybrid approaches under different training set sizes (budget strategies). Our goal is to evaluate the performance of these AL algorithms in reducing annotation costs while maintaining high model performance.

Results show that in the case of using CIFAR-10 and ResNet-18 model, there is no significant difference between algorithms. This highlights how important the conditions of the query selections are when determining the efficiency of AL techniques.

## Implementation Details

### Active Learning Algorithms
- Random sampling (baseline)
- Uncertainty-based sampling
- Typiclust sampling
- Margin sampling
- Entropy sampling
- BADGE (Batch Active learning by Diverse Gradient Embeddings)
- Core-set sampling

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running the Simulation
```bash
python active_learning_simulation.py
```

Or use the Jupyter notebook for interactive exploration:
```bash
jupyter notebook simulation_explainer_notebook.ipynb
```

## Results
The simulation results are stored in the `run_results/` directory and include:
- Test accuracy vs. labeled dataset size
- Training and AL algorithm runtimes
- Cross-seed performance comparisons
- Budget strategy effectiveness

## License
This project is licensed under the MIT License - see the 

LICENSE

 file for details.
