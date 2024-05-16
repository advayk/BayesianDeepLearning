## Overview

This repository contains code to demonstrate how to load training and testing data, particularly focusing on applying Gaussian blur to the testing data. The testing data is categorized into three levels of blur: light, medium, and heavy. Below is a brief summary of the provided code.

## Code Description

The provided code demonstrates how to visualize the testing data with different levels of Gaussian blur. Specifically, it:

1. **Imports Libraries**: Imports the necessary libraries (`matplotlib.pyplot` and `torchvision.transforms`).
2. **Loads Blurred Images**: Selects specific images from the `blurred_testsets` dictionary, which contains datasets with different blur levels (light, medium, heavy).
3. **Converts Images**: Converts the selected images from tensors to PIL images for compatibility with plotting functions.
4. **Plots Images**: Creates a subplot to display the images side by side, each labeled with its respective blur level (Light Blur, Medium Blur, Heavy Blur).

## Visualization

The blurred images are plotted to provide a visual comparison of the different blur levels. This helps in understanding the impact of Gaussian blur on the images.

## Conclusion

This code snippet is part of a larger project focused on data preprocessing and visualization. By applying Gaussian blur to the testing data, we can evaluate how different levels of blur affect our models. For more details and comprehensive examples, please refer to the main notebook in this repository.
