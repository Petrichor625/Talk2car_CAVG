# 🚗 **GPT-4 Enhanced Multimodal Grounding for Autonomous Driving: Leveraging Cross-Modal Attention with Large Language Models**
## 🏆 **High Impact Research**

This repository contains the official implementation of **GPT-4 Enhanced Multimodal Grounding for Autonomous Driving: Leveraging Cross-Modal Attention with Large Language Models**, published in the journal *Communications in Transportation Research*. 

🔥 **Essential Science Indicators High-Citation Paper** — ranked in the top **1% of most-cited papers** in the field.

---


## 📖 Overview

Welcome to the official repository for **GPT-4 Enhanced Multimodal Grounding for Autonomous Driving: Leveraging Cross-Modal Attention with Large Language Models**. This project introduces a novel approach using GPT-4 to enhance autonomous vehicle (AV) systems with a human-centric multimodal grounding model. The CAVG model combines text, visual, and contextual understanding for improved intent prediction in complex driving scenarios.



## ✨ Highlights

- **Hybrid Strategy for Contextual Analysis**  
  A pioneering hybrid approach for advanced image-text context analysis tailored to autonomous vehicle command grounding.
  
- **Cross-Modal Attention Mechanism**  
  A unique cross-modal attention mechanism for deriving nuanced human-AV interactions from multimodal inputs.

- **Large Language Model Integration**  
  Leverages GPT-4 for effective embedding and interpretation of emotional nuances in human commands.

- **Robustness in Diverse Scenarios**  
  Demonstrates exceptional performance across challenging traffic environments, validated extensively on the Talk2Car dataset.



## 📜 Abstract

Navigating complex commands in a visual context is a core challenge for autonomous vehicles (AVs). Our **Context-Aware Visual Grounding (CAVG)** model employs an advanced encoder-decoder framework to address this challenge. Integrating five specialized encoders—Text, Image, Context, Cross-Modal, and Multimodal—the CAVG model leverages GPT-4’s capabilities to capture human intent and emotional undertones. The model's architecture includes multi-head cross-modal attention and a Region-Specific Dynamic (RSD) layer for enhanced context interpretation, making it resilient across diverse and challenging real-world traffic scenarios. Evaluations on the Talk2Car dataset show that CAVG outperforms existing models in accuracy and efficiency, excelling with limited training data and proving its potential for practical AV applications.



## 🧠 Framework

**Model Architecture**  
- **Text Encoder & Emotion Encoder**: Generate a text and an emotion vector from commands.
- **Vision Encoder**: Processes images into Regions of Interest (RoIs).
- **Context Encoder & Cross-Modal Encoder**: Enrich RoIs contextually and merge using multi-head cross-modal attention.
- **Multimodal Decoder**: Scores each region’s likelihood and selects the top-\(k\) regions matching the command semantics.



<img src="https://github.com/Petrichor625/Talk2car_CAVG/blob/main/Figure/framework.png" alt="Framework Diagram" width="800"/>


## To-do List

###### **Note**

- [x] [2023.10.25] Creating the repository for CVAG 
- [x] [2023.11.05] Open source CAVG code
- [x] [2023.11.28] Update the Readme
- [x] [2023.11.28] Update project code
- [x] [2024.11.11] Update the Readme



## 🛠️ Requirements

### Environment

- **Operating System**: Ubuntu 22.04
- **CUDA Version**: 11.7

### Setup Instructions

1. **Create Conda Environment**  
   ```bash
   conda create --name CAVG python=3.7
   conda activate CAVG
```

2. **Install PyTorch with CUDA 11.7**  

   ```bash
   conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
   ```

3. **Install Additional Requirements**  

   ```bash
   pip install -r requirements.txt
   ```



## 📂 Talk2Car Dataset

Experiments are conducted using the **Talk2Car** dataset. If you use this dataset, please cite the original paper:

```bibtex
Thierry Deruyttere, Simon Vandenhende, Dusan Grujicic, Luc Van Gool, Marie-Francine Moens:
Talk2Car: Taking Control of Your Self-Driving Car. EMNLP 2019
```

### Dataset Download Instructions

1. **Activate Environment and Install `gdown`**  

   ```bash
   conda activate CAVG
   pip install gdown
   ```

2. **Download Talk2Car Images**  

   ```bash
   gdown --id 1bhcdej7IFj5GqfvXGrHGPk2Knxe77pek
   ```

3. **Organize Images**  

   ```bash
   unzip imgs.zip && mv imgs/ ./data/images
   rm imgs.zip
   ```



## 🏋️‍♂️ Training

To start training the CAVG model with the Talk2Car dataset, run:

```bash
bash talk2car/script/train.sh 
```



## 📊 Evaluation

To evaluate the model's performance, execute:

```bash
bash talk2car/script/test.sh
```



## 🔍 Prediction

During the prediction phase on the Talk2Car dataset, bounding boxes are generated to assess the model's spatial query understanding. To begin predictions, run:

```bash
bash talk2car/script/prediction.sh
```



## 🎨 Qualitative Results

**Performance Comparison**  
Ground truth bounding boxes are in blue, while CAVG output boxes are in red. Commands associated with each scenario are displayed for context.
![image](https://github.com/Petrichor625/Talk2car_CAVG/blob/main/Figure/talk2car(1)%20(2).png)

**Challenging Scenes**  
Examples from scenes with limited visibility, ambiguous commands, and multiple agents.

![image](https://github.com/Petrichor625/Talk2car_CAVG/blob/main/Figure/carner_case_03.png)


## 🏆 Leaderboard

Models on Talk2Car are evaluated by Intersection over Union (IoU) of predicted and ground truth bounding boxes with a threshold of 0.5 (AP50). We welcome pull requests with new results!

| Model                                                        | AP50 (IoU<sub>0.5</sub>) | Code                                                         |
| ------------------------------------------------------------ | ------------------------ | ------------------------------------------------------------ |
| [STACK-NMN](https://arxiv.org/pdf/1807.08556.pdf)            | 33.71                    |                                                              |
| [SCRC](https://arxiv.org/abs/1511.04164)                     | 38.7                     |                                                              |
| [OSM](https://arxiv.org/pdf/1406.5679.pdf)                   | 35.31                    |                                                              |
| [Bi-Directional retr.](https://arxiv.org/abs/2004.13822)     | 44.1                     |                                                              |
| [MAC](https://arxiv.org/abs/1803.03067)                      | 50.51                    |                                                              |
| [MSRR](https://arxiv.org/abs/2003.08717)                     | 60.04                    |                                                              |
| [VL-Bert (Base)](https://arxiv.org/abs/1908.08530)           | 63.1                     | [Code](https://github.com/ThierryDeruyttere/VL-BERT-Talk2Car) |
| [AttnGrounder](https://arxiv.org/abs/2009.05684)             | 63.3                     | [Code](https://github.com/i-m-vivek/AttnGrounder)            |
| [ASSMR](https://link.springer.com/chapter/10.1007/978-3-030-66096-3_5) | 66.0                     |                                                              |
| [CMSVG](https://arxiv.org/abs/2009.06066)                    | 68.6                     | [Code](https://github.com/niveditarufus/CMSVG)               |
| [Vilbert (Base)](https://arxiv.org/abs/1908.02265)           | 68.9                     | [Code](https://github.com/ThierryDeruyttere/vilbert-Talk2car) |
| [CMRT](https://link.springer.com/chapter/10.1007/978-3-030-66096-3_3) | 69.1                     |                                                              |
| [Sentence-BERT+FCOS3D](https://www.aaai.org/AAAI22Papers/AAAI-8858.GrujicicD.pdf) | 70.1                     |                                                              |
| [Stacked VLBert](https://link.springer.com/chapter/10.1007/978-3-030-66096-3_2) | 71.0                     |                                                              |
| [FA](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9961196) | 73.51                    |                                                              |
| **CAVG (Ours)**                                              | 74.55                    | [Code](https://github.com/Petrichor625/Talk2car_CAVG)        |

You can find the full Talk2Car leaderboard [here](https://github.com/talk2car/Talk2Car/blob/master/leaderboard.md).

---

## 📑 Citation

If you find our work useful, please consider citing:

``
**BibTex**
@article{LIAO2024100116,
title = {GPT-4 enhanced multimodal grounding for autonomous driving: Leveraging cross-modal attention with large language models},
journal = {Communications in Transportation Research},
volume = {4},
pages = {100116},
year = {2024},
issn = {2772-4247},
doi = {https://doi.org/10.1016/j.comm

tr.2023.100116},
url = {https://www.sciencedirect.com/science/article/pii/S2772424723000276},
author = {Haicheng Liao and Huanming Shen and Zhenning Li and Chengyue Wang and Guofa Li and Yiming Bie and Chengzhong Xu},
keywords = {Autonomous driving, Visual grounding, Cross-modal attention, Large language models, Human-machine interaction}
}
```
GPT-4 Enhanced Multimodal Grounding for Autonomous Driving: Leveraging Cross-Modal Attention with Large Language Models accepted by the journal _Communications in Transportation Research_.
Thank you for exploring CAVG! Your support and feedback are highly appreciated.





