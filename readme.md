# Landfill Detection using Satellite Imagery

A machine learning pipeline for detecting the existence of landfills in a given area of Sentinel-2 L2A satellite image data.

The guidance of the project can be found in **docs\ML_Engineer_CGG_Incubator.pdf**. In this page, I will go through the project and answer questions.

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Data Collection](#data-collection)
- [Modeling](#modeling)
- [ML Pipeline](#ml-pipeline)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Future Work](#future-work)

## Introduction

This repository contains an end-to-end machine learning pipeline designed to detect the existence of a landfill in one area from a given satellite image. The pipeline mainly consists of the following parts: data collection, model training, model evaluation, predictions, and deployment.

## Prerequisites

- Python 3.10.6 
- torch 2.0.0+cu118
- torchvision 0.15.1+cu118
- GPU: Nvidia Geforce GTX 1060

## Data Collection

The data collection is implemented in the `SatelliteDataset` Class in **src\dataset.py**. Based on the parameters of the dataset \(collections, datetime, geojson file, etc. \), the images will be downloaded in real-time during training or testing. The next time, if the dataset detects the downloaded files, it will directly read it to save time.

Answering questions: 

> What are the sizes of these images? Do you need to do some processing for the model building?

By directly running the dataset module

```bash
python CGG_test_landfill_prediction/src/dataset.py
```

We can find out that all training and test images are nearly of size **150x150**.

> With the help of [this description](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a#overview), can you tell the real-world resolution of downloaded images? \(optional\) 

According to the "GSD" \(Ground Sample Distance\) column of the **Spectral Bands** table on the page, we can see that the real-world resolution of downloaded images, when considering the "visual" asset, is 10 meters. And for other bands, like B05-B07 \(Vegetation red edge\), the "GSD" is 20m. For B01\(Coastal aerosol\) and B09\(Water vapo\), their resolution is 60m.

> Any other insight that you can draw from the data? \(optional\)

- The dataset offers a variety of asset types, not just the spectral bands. These include aerosol optical thickness, scene classification map, vegetation red edge, and water vapour, which can provide additional context or information when analyzing the existence of landfills.

- This is a continuously updated database, so it also provides a large amount of time series data, which is very valuable for many other projects.

## Modeling

Based on the requirements of this project, I plan to choose a simple image classification model. **ResNet18** seems to fit the bill. The ResNet series has always been a commonly used model in production, with good performance, simple operators, solid pre-trained support, and compatibility with PyTorch. Additionally, it's within the computational capacity that my laptop can handle.

Answering questions:

> How do you evaluate your model?

As the problem is a binary classification, the choice of the threshold will directly influence the result. So, I first chose the **AUC**\(Area Under the Curve\) to evaluate my model. Along with the AUC metric, the best threshold will be calculated by using Youden's J statistic, and then other metrics are applied, such as accuracy, precision, recall, and the F1 score, to evaluate my model. The metrics are implemented in **src\metrics.py**

> What do you think about the performance that you got with your model? Any reason?

> Any suggestions to improve the performance?

> What could impact the performance of a machine learning model?

## ML Pipeline

## Usage

## Evaluation Metrics

## Future Work

