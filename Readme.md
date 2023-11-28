#  CAVG

## Overview

This repository contains the official implementation of **GPT-4 Enhanced Multimodal Grounding for Autonomous Driving: Leveraging Cross-Modal Attention with Large Language Models.**


## Highlights

- Developing a pioneering hybrid strategy for advanced image-text context analysis in autonomous vehicle command grounding.
- Introducing a novel cross-modal attention mechanism, uniquely tailored to derive insightful human-AV interaction from multi-modal inputs.
- Leveraging the sophisticated capabilities of the large language model GPT-4 for effective embedding and nuanced interpretation of human emotions in commander commands.
- Demonstrating exceptional robustness and adaptability of our model across a spectrum of challenging traffic scenarios, validated extensively on the Talk2Car dataset.



## Abstract

In the field of autonomous vehicles (AVs), accurately discerning commander intent and executing linguistic commands within a visual context presents a significant challenge. This paper introduces a sophisticated encoder-decoder framework, developed to address visual grounding in AVs. Our Context-Aware Visual Grounding (CAVG) model is an advanced system that integrates five core encoders—Text, Image, Context, and Cross-Modal—with a Multimodal decoder. This integration enables the CAVG model to adeptly capture contextual semantics and to learn human emotional features, augmented by state-of-the-art Large Language Models (LLMs) including GPT-4. The architecture of CAVG is reinforced by the implementation of multi-head cross-modal attention mechanisms and a Region-Specific Dynamic (RSD) layer for attention modulation. This architectural design enables the model to efficiently process and interpret a range of cross-modal inputs, yielding a comprehensive understanding of the correlation between verbal commands and corresponding visual scenes. Empirical evaluations on the Talk2Car dataset, a real-world benchmark, demonstrate that CAVG establishes new standards in prediction accuracy and operational efficiency. Notably, the model exhibits exceptional performance even with limited training data, ranging from 50% to 75% of the full dataset. This feature highlights its effectiveness and potential for deployment in practical AV applications. Moreover, CAVG has shown remarkable robustness and adaptability in challenging scenarios, including long-text command interpretation, low-light conditions, ambiguous command contexts, inclement weather conditions, and densely populated urban environments.


## Framework
![image]([https://github.com/Petrichor625/Talk2car_CAVG/blob/main/Figure/talk2car(1)%20(2).png](https://github.com/Petrichor625/Talk2car_CAVG/blob/main/Figure/framework_7%20(1).png))
The Text Encoder and the Emotion Encoder generate a text vector and an emotion vector, respectively, from the given command, while the Vision Encoder divides the input image into \(N\) RoIs, each represented by a vision vector. These vectors are contextually enriched by a context encoder and then merged by a Cross-Modal Encoder using multi-head cross-modal attention. The multimodal decoder calculates likelihood scores for each region and selects the top-\(k\) regions that best match the semantics of the command. The final prediction is based on this fusion.


## To-do List

###### **Note**

- [x] [2023.10.25] Creating the repository for CVAG 
- [x] [2023.11.05] Open source CAVG code
- [x] [2023.11.28] Update Readme
- [x] [2023.11.28] Update project code



## Requirements

### Environment

- **Operating System**: Ubuntu 22.04
- **CUDA Version**: 11.7

### Setting Up 

1.**Creating the Conda Environment for CAVG**: 

For optimal use of the CAVG, follow these setup guidelines:

```
conda create -name CAVG python=3.7
conda activate CAVG
```

2.**Installing PyTorch**:

 Install PyTorch and associated libraries compatible with CUDA 11.7:

```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

**Installing Additional Requirements**: 

Complete the environment setup by installing the necessary packages from `requirements.txt`:

```
pip install -r requirements.txt
```



## Talk2car Dataset

Experiments were conducted using the Talk2Car dataset. Should you utilize this dataset in your work, please ensure to cite the original paper.

```
Thierry Deruyttere, Simon Vandenhende, Dusan Grujicic, Luc Van Gool, Marie-Francine Moens:
Talk2Car: Taking Control of Your Self-Driving Car. EMNLP 2019
```

#### Downloading the Dataset

1. Activate the CAVG environment and install `gdown` for downloading the dataset:

```
conda activate CAVG
pip install gdown
```

2.Download the Talk2Car images:

```
gdown --id 1bhcdej7IFj5GqfvXGrHGPk2Knxe77pek
```

3.Unzip and organize the images:

```
unzip imgs.zip && mv imgs/ ./data/images
rm imgs.zip
```



## Train

To begin training your CAVG model with the Talk2Car dataset, you can easily start with the provided script. Training is a crucial step to ensure your model accurately understands and processes the dataset.

Simply run the following command in your terminal:

```
bash talk2car/script/train.sh 
```



## Evaluation

Execute the following command to start the evaluation process:

```
bash talk2car/script/test.sh
```



## Prediction

The prediction phase in the Talk2Car dataset plays a critical role. It involves generating bounding boxes for each command as part of the object referral task. This step is vital for assessing the effectiveness of your model in understanding and responding to spatial queries within the dataset.

Execute the following command in your terminal to initiate the prediction process:

```
bash talk2car/script/prediction.sh
```



## Leadboard

Here is where you can find the [leaderboard]([Talk2Car/leaderboard.md at master · talk2car/Talk2Car (github.com)](https://github.com/talk2car/Talk2Car/blob/master/leaderboard.md)) for the Talk2Car.





## Qualitative Results
![image](https://github.com/Petrichor625/Talk2car_CAVG/blob/main/Figure/talk2car(1)%20(2).png)
Comparative Visualization of Model Performance on the Talk2Car Dataset. Ground truth bounding boxes are depicted in blue, while output bounding boxes of CAVG are highlighted in red. A natural language command associated with each visual scenario is also displayed below the image for context.


![image](https://github.com/Petrichor625/Talk2car_CAVG/blob/main/Figure/carner_case_03.png)
Comparative Visualization of Model Performance on Challenging Scenes. The challenging scenes include those with limited visibility, ambiguous commands, and scenes with multiple agents. Ground truth bounding boxes are depicted in blue, while output bounding boxes of CAVG are highlighted in red. A natural language command associated with each visual scenario is also displayed below the image for context.



## Citation

GPT-4 Enhanced Multimodal Grounding for Autonomous Driving: Leveraging Cross-Modal Attention with Large Language Models (Camera-ready)
