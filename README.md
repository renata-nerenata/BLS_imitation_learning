# BLS_imitation_learning

[![Actions Status](https://github.com/renata-nerenata/BLS_imitation_learning/workflows/Linter/badge.svg)](https://github.com/renata-nerenata/BLS_imitation_learning/actions)
[![Test Coverage](https://api.codeclimate.com/v1/badges/a48adcf1d95882daed37/test_coverage)](https://codeclimate.com/github/renata-nerenata/BLS_imitation_learning/test_coverage)

This repository represents the code that is described our project for extracting player strategies in Borderland Science, an online science discovery game, which, in its turn, includes a multi-sequence alignment game. We approached this task with Behavioural Cloning using two types of neural networks: Transformers and Fully Convolutional networks. Previous research has largely focused on classical Reinforcement Learning (RL) methods and missed the opportunity to use the real experience of people, which often surpasses existing methods of multiple sequence alignments. Our results show that these approach can match human performance by learning basic dynamics and rules.

## Installation
- Get Frame Interpolation source codes
```console
git clone
```
- Optionally, Docker base image
```console
docker build . -t <image_name>
docker run -it <image_name>
```
- Install dependencies
```console
pip install -r requirements.txt
```

## Pre-trained models

We trained two models: Transformers and Fully Convolutional networks. Take on of the thow labels 'transformer' or 'FCN' for model_type

```console
python inference.py --puzzle <puzzle> --model <model_type>
```
--ADD DEMO

## Datasets

Each puzzle is a sample of data where the sequences are represented as a list of strings, and the player's step is coded according to the sequence number and the letter, which is the index number where the gap was inserted. The difficulty of the presented puzzles varies from 1 to 9, but more than 50 percent of the puzzles have a difficulty less than 3. 

## Evaluation
To run an evaluation, simply pass the configuration file of the desired evaluation dataset. 
```console
python metrics/metrics.py --puzzle_real <puzzle_real> --puzzle_predict <puzzle_predict>
```
