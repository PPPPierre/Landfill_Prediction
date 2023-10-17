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

1. What are the sizes of these images? Do you need to do some processing for the model building?

    By directly running the dataset module

    ```bash
    python CGG_test_landfill_prediction/src/dataset.py
    ```

    We can find out that all training and test images are nearly of size **150x150**.

2. With the help of [this description](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a#overview), can you tell the real-world resolution of downloaded images? \(optional\) 

    According to the "GSD" \(Ground Sample Distance\) column of the **Spectral Bands** table on the page, we can see that the real-world resolution of downloaded images, when considering the "visual" asset, is 10 meters. And for other bands, like B05-B07 \(Vegetation red edge\), the "GSD" is 20m. For B01\(Coastal aerosol\) and B09\(Water vapo\), their resolution is 60m.

3. Any other insight that you can draw from the data? \(optional\)

    - The dataset offers a variety of asset types, not just the spectral bands. These include aerosol optical thickness, scene classification map, vegetation red edge, and water vapour, which can provide additional context or information when analyzing the existence of landfills.

    - This is a continuously updated database, so it also provides a large amount of time series data, which is very valuable for many other projects.

## Modeling

Based on the requirements of this project, I plan to choose a simple image classification model. **ResNet18** seems to fit the bill. The ResNet series has always been a commonly used model in production, with good performance, simple operators, solid pre-trained support, and compatibility with PyTorch. Additionally, it's within the computational capacity that my laptop can handle.

Answering questions:

4. How do you evaluate your model?

    As the problem is a binary classification, the choice of the threshold will directly influence the result. So, I first chose the **AUC**\(Area Under the Curve\) to evaluate my model. Along with the AUC metric, the best threshold will be calculated by using Youden's J statistic, and then other metrics are applied, such as accuracy, precision, recall, and the F1 score, to evaluate my model. The metrics are implemented in **src\metrics.py**

5. What do you think about the performance that you got with your model? Any reason?

    Result on test dataset:

    ```
    AUC: 0.778, accuracy: 0.833, precision: 1.000, recall: 0.667, f1_score: 0.800
    ```

    Given the metrics, the model showcases a good AUC of 77.8%, an accuracy rate of 83.3%, a precision of 100%, a recall of 66.7%, and a f1_score of 80%. However, despite these good results, the limited amount of data raises my concerns regarding the reliability of these metrics and the risk of overfitting. It's essential to note that such performance while appearing robust, may not guarantee the model's generalization to unseen data. Gathering more data can help ensure a more dependable performance and reduce the potential for overfitting.

6. Any suggestions to improve the performance?

    To improve performance, here I suggest considering the following:

    - Acquiring More Data: Gathering more data can significantly improve model performance.
    - Data Augmentation: In addition to traditional image data augmentation methods, more targeted data augmentation approaches can be designed based on the characteristics of this dataset: By using the training data's labels, one can randomly select satellite images that contains the training data range and then augment them through random cropping and resizing.

7. What could impact the performance of a machine learning model?

    A machine learning model can be, but is not limited to be, influenced by the following factors:

    - Quality of Data: Noise or inaccuracies in data can lead to poor model performance.
    - Quantity of Data: Insufficient data can prevent the model from generalizing well.
    - Model Complexity: Overly complex models might overfit to the training data, while overly simple models might underfit.
    - Hyperparameters: Incorrectly tuned hyperparameters can degrade performance.
    - Feature Engineering: The way data is represented and organized can significantly influence results.
    - Training Epochs: Inadequate training time can lead to underfitting.

## ML Pipeline

The goal here is to build a reproducible pipeline.

Based on the previous steps of data downloading and modeling, to deploy your model in production, develop a machine learning pipeline that:

1. Reads the raw data and downloads the satellite images

    In **src\dataset.py**, I extended PyTorch's `Dataset` by implementing the `SatelliteDataset` class, responsible for reading raw data, downloading images and applying data augmentation techniques. During image retrieval, the dataset temporarily stores pictures locally, eliminating the need for re-downloading throughout subsequent training steps. 

2. Trains the model

    The training procedures are articulated directly in the **src\train.py** file, executed via the `train` method. Additionally, in the root directory, I established a parameter-driven `main_train.py` script. By inputting a well-crafted `config.yaml` file as an argument, users can initiate the training process seamlessly.

3. Evaluate the model with the metrics

    The model evaluation steps are encapsulated in the **src\evaluate.py** file and operationalized through the `evaluate` method. This function is invoked after each epoch during the training phase, logging the performance and progression of the model systematically.

4. Make predictions on new locations and show the result

    The methodology for predicting new data via the trained model is contained in the **src\predict.py** file, utilizing the `predict` method. It is noteworthy that for excessively large images, an approach dubbed **crop with overlap** is employed. This technique involves segmenting the input image into several patches, conducting predictions for each fragment, and summarize the outcomes. This is primarily due to two reasons: 1. Landfills exhibit limited distinctive features within images, and resizing larger pictures could result in reduced resolution, adversely impacting model accuracy. 2. The model's training dataset is collected with standardized dimensions. Even though data augmentation strategies might enhance model generalization across varying resolutions, maintaining target detection within a specific resolution range is imperative for meaningful analysis.

5. Add any bricks that you think are useful to this pipeline

    I personally integrated two significant features into this pipeline:

    - YAML Config Mechanism

        From data handling and model training to evaluation, every aspect of the workflow is driven by a YAML-configured file. Users can define the entire training sequence within a single file, offering a structured format that not only facilitates input validation but also ensures efficient compatibility with other interfaces. This approach is particularly advantageous for future development based on this framework.

    - Register Mechanism

        Modules requiring extension, including data augmentation, models, optimizers, schedulers, loss functions, and metrics, can be actualized using the register mechanism. While I haven't implemented every possibility, the core concept revolves around enhancing code reusability through this mechanism. Users specify the necessary modules within the config file, which are then instantiated directly by a register factory. Future extensions necessitate only new code additions, significantly minimizing alterations to existing structures, thereby fostering more manageable maintenance and scalability of the project.

6. Propose an interface so that this pipeline can be run easily. (optional)

7. Propose a solution for monitoring in production (optional)

## Usage

### For start training script
1. prepare train and test dataset in geojson format: `data\raw\train.geojson` `data\raw\test.geojson`
2. prepare a config file in YAML format, refer to `configs\resnet18_train.yaml`
3. run the training script, pass the config file path in the project as argument
```bash
python ./main.py --config ./configs/resnet18_train.yaml
```
4. check the result directory, the log will be saved in `run.log`, the model weights will be saved in `model` directory.

### For start a prediction
1. prepare predict data file in geojson format: `data\raw\pred.geojson`
2. prepare a config file in YAML format, refer to `configs\resnet18_pred.yaml`, change the `model` in `model` to the reletive path of the checkpoint trained.
3. run the prediction script, pass the config file path in the project as argument
```bash
python ./main.py --config ./configs/resnet18_pred.yaml
```
4. check the result directory, the results will be saved in `result.geojson`

### For launch a prediction service
1. run the script file, start the Flask app
```bash
python ./main_infer_service_app.py 
```
2. run the client script `scripts\inference_client.py` to test the service
```bash
python ./scripts/inference_client.py
```

## Future Work

