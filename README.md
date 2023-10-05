# Project_MLops_DTU
Fruit image classification using Vision Transformer 
===================================================

This repository contains the project work of our group for the DTU special course [Machine Learning Operations](https://kurser.dtu.dk/course/02476) for the Autumn semester 2023.

Team 31 members:
- Theodoros Loukis s223526
- Ioannis Louvis s222556
- Ioannis Karampinis s222559
- Elena Muniz s2134579


### Overall goal
The goal of the project is to fine tune a deep learning model based on [Vision Transformer (ViT)](https://huggingface.co/docs/transformers/model_doc/vit) that classifies the quality of fruits by their image.
 

### Framework
We plan to use the tranformer framework from [Huggingface](https://huggingface.co/). Specifically, use the Vision Transformer based on the paper:
[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).


### How to include the framework
We want to use the Transformers framework that includes many pretrained models, which wil intend to use in order to transfer, learn and train our classification model in the dataset bellow.


### Dataset
We plan to use the [FRUIT CLASSIFICATION](https://www.kaggle.com/datasets/shashwatwork/fruitnet-indian-fruits-dataset-with-quality) dataset from Kaggle. This is a dataset that contains a total of more than 14700 high quality fruit images of 6 different classes of fruits i.e. apple, banana, guava, lime, orange, and pomegranate. Our goal is to classify them to different classes based on their quality:
- Good.
- Bad.
- Mixed.

### Deep learning models
We expect to use the [Vision Transformer (ViT)](https://huggingface.co/docs/transformers/model_doc/vit) model, which is a deep learning model and is a transformer that is targeted at vision processing tasks such as image recognition. 
We might as well also try the [BERT Pre-Training of Image Transformers (BEiT)](https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/beit) and/or [Data-efficient Image Transformers (DeiT)](https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/deit) models, which are follow-up works on the original ViT model.
