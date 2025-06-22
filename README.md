# Codes of Improving Multimodal Learning Balance and Sufficiency through Data Remixing

Here is the PyTorch implementation of ''*Improving Multimodal Learning Balance and Sufficiency through Data Remixing*'', which aims to alleviate modality imbalance and modality clash in multimodal learning by decoupling the multimodal input and reassembling by batch. Please refer to our [ICML 2025 paper](https://arxiv.org/abs/2506.11550) for more details.

**Paper Title: "Improving Multimodal Learning Balance and Sufficiency through Data Remixing"**

**Authors: Xiaoyu Ma, Hao Chen, Yongjian Deng**

**Accepted by: 2025 Forty-Second International Conference on Machine Learning(ICML 2025)**

## Motivation and Method

<img src="./motivation.png" style="zoom:50%;" />

In this work, we address a long-standing but often overlooked problem in the field of balanced multimodal learning, namely **Modality Clash**. This issue arises from differences in the optimization directions of different modalities, which can lead to insufficient learning across all modalities:

+ **Modality Imbalance** is **unidirectional**, which refers to a scenario where a strong modality dictates the learning process, preventing **other modalities** from being sufficiently trained. 

+ **Modality Clash** is **bidirectional**, which describes interference between modalities. *Even if modality balance is achieved*, differences between modalities may still lead to insufficient learning across **all modalities**.

The pipeline of Data Remixing is as follows:

<img src="./pipeline.png" style="zoom:50%;" />

## Code Instruction

+ Our code will be released in a period of time. If you need it soon, please contact us: xiaoyuma.kb@gmail.com

