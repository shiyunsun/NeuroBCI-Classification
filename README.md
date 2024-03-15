# EEG Signal Classification with Deep Learning

## Team Members

- Shiyun Sun
- Ryo James Matsumoto

## Abstract

This project explores the classification of EEG signals using deep learning models including CNN, LSTM, and Transformer architectures. The focus is on understanding the impact of attention mechanisms and the behavior of these models across different sequence lengths. Our findings highlight the Transformer model's superior performance, achieving a peak test accuracy of approximately 76% to 77% with significantly fewer parameters and training feasibility on CPUs.

## Introduction

The project addresses the classification of EEG signals, emphasizing the use of recurrent neural networks (RNNs), Long Short-Term Memory (LSTM) networks, and Transformer models. Given the limitations of computational resources, the project also explores the effectiveness of attention mechanisms in improving model performance.

## Dataset

The EEG dataset used in this project comes from the BCI Competition IV, specifically Dataset 2A, consisting of EEG data from 9 subjects performing motor imagery tasks. Each subject's data includes 22 EEG channels over 1000 time bins for multiple trials, with each trial corresponding to one of four imagined movements: left hand, right hand, both feet, or tongue.

### Accessing the Dataset

The dataset is made publicly available through the BCI Competition website: https://www.bbci.de/competition/iv/. We processed the data to be easily loadable for analysis, removing trials with NaN values and normalizing the data for each subject.


## Results

Refer to [model_train.ipynb](/model_train.ipynb).
