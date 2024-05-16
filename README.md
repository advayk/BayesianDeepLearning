# Bayesian Deep Learning and a Probabilistic Perspective of Generalization

## 3.1 Introduction

This project is a re-implementation of the paper titled "Bayesian Deep Learning and a Probabilistic Perspective of Generalization" by Andrew Gordon Wilson and Pavel Izmailov, presented at NeurIPS 2020. The main contribution of the paper is the introduction of Bayesian marginalization in deep neural networks to improve accuracy and calibration. The method emphasizes using multiple weight configurations, further enhanced by Stochastic Weight Averaging-Gaussian (SWAG) to improve predictive distributions.

## 3.2 Chosen Result

We aimed to reproduce the negative log likelihood performance of Deep Ensembles, MultiSWAG, and MultiSWA on the CIFAR-10 dataset under varying intensities of Gaussian blur corruption. This result demonstrates the effectiveness of MultiSWAG in capturing uncertainty and improving generalization compared to traditional deep ensembles. The relevant figure from the original paper illustrates the comparative performance of these methods under different conditions of data corruption.

<div style="text-align: center;">
  <img src="results/deep_ensembles_vs_multiSWAG.png" alt="Deep Ensembles vs MultiSWAG" style="width: 50%;">
  <p><em>Figure 1: Comparative performance of Deep Ensembles, MultiSWAG, and MultiSWA under varying Gaussian blur corruption on CIFAR-10.</em></p>
</div>


## 3.3 Re-implementation Details

### Approach

1. **Model**: ResNet18 from torchvision.
2. **Dataset**: CIFAR-10, which consists of 60,000 32x32 color images in 10 classes (50,000 training images and 10,000 test images).
3. **Evaluation**: Performance assessed on three levels of image blurring: light, medium, and heavy.
4. **Ensembling**: 10 models were ensembled to evaluate their negative log likelihood.

### Running the Code

1. **Dependencies**: 
   - Python 3.8+
   - PyTorch
   - torchvision
   - numpy
   - matplotlib

2. **Instructions**:
   See SWAG.ipynb and run all the cells. 

### Computational Resources

- GPU is highly recommended for training due to the computational intensity of deep learning models.
- The implementation was tested on NVIDIA GTX 1080 Ti.

## 3.4 Results and Analysis

### Re-implementation Results

The re-implementation showed the following results:

- **MultiSWAG**: Achieved the best performance in terms of negative log likelihood under varying Gaussian blur intensities.
- **Deep Ensembles**: Provided robustness but were outperformed by MultiSWAG in terms of capturing uncertainty.

<div style="text-align: center;">
  <img src="results/re-implementation_deep_ensembles_vs_multiSWAG.png" alt="Deep Ensembles vs MultiSWAG" style="width: 30%;">
  <p><em>Figure 2: Our re-implementation of the comparative performance of Deep Ensembles, MultiSWAG, and MultiSWA under varying Gaussian blur corruption on CIFAR-10.</em></p>
</div>

### Discrepancies and Challenges

- **Discrepancies**: There were minor differences in scaling for negative log likelihood compared to the original paper. This was hypothesized to be due to variations in hyperparameter settings and computational constraints.
- **Challenges**: Implementing SWAG was the most challenging aspect due to its complexity and the need for precise tuning.

### Analysis

Our results align with the paper's findings that Bayesian approaches, especially MultiSWAG, provide superior generalization by effectively capturing model uncertainty. This is crucial in real-world applications where uncertainty estimation can significantly impact decision-making.

## 3.5 Conclusion and Future Work

### Key Takeaways

- Bayesian Deep Learning offers a robust framework for improving model generalization and uncertainty estimation.
- MultiSWAG is an effective method for approximating Bayesian model averaging without significant computational overhead.

### Future Directions

- **Scalability**: Explore scalable implementations for larger datasets and deeper models.
- **Extensions**: Investigate other forms of Bayesian approximations and their impact on different types of neural network architectures.

## 3.6 References

- Wilson, A. G., & Izmailov, P. (2020). Bayesian Deep Learning and a Probabilistic Perspective of Generalization. Advances in Neural Information Processing Systems, 33.
- Izmailov, P., et al. (2019). Averaging Weights Leads to Wider Optima and Better Generalization. arXiv:1803.05407.
- Maddox, W., et al. (2019). A Simple Baseline for Bayesian Uncertainty in Deep Learning. arXiv:1902.02476.
