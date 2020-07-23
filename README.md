# Cats VS Dogs

A CNN to distinguish between cats and dogs. <br>

Training set of 2000 images, 1000 images of cats and dogs each.

## Accuracies

**100 epochs**

>With Augmentation

<p>Gives a training accuracy of 89% and a validation accuracy of 87.1%</p>

<p float="left">
    <img src="models/100epochs/model-2-acc-aug.png" width="300" height="300">
    <img src="models/100epochs/model-2-loss-aug.png" width="300" height="300">
    <img src="models/100epochs/model-2-val-acc-aug.png" width="300" height="300">
    <img src="models/100epochs/model-2-val-loss-aug.png" width="300" height="300">
</p>


>Without Augmentation

<p>Give a training accuracy of 99% and validation accuracy of 76%</p>

<p float="left">
    <img src="models/100epochs/model-2-acc-noaug.png" width="300" height="300">
    <img src="models/100epochs/model-2-loss-noaug.png" width="300" height="300">
</p>


**50 epochs**

>Without Augmentation

<p>Gives a training accuracy of 85% and validation accuracy of 84%</p>

<p float="left">
    <img src="models/50epochs/model-2-acc.png" width="300" height="300">
    <img src="models/50epochs/model-2-loss.png" width="300" height="300">
    <img src="models/50epochs/model-2-val-acc.png" width="300" height="300">
    <img src="models/50epochs/model-2-val-loss.png" width="300" height="300">
</p>


> Trained on Google Colab

## Code

Cats_vs_Dogs_model2.ipynb (50 epochs)
