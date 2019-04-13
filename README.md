# DEmoClassi
DEmoClassi stands for **D**emographic (age, gender, race) and **Emo**tions 
(happy, sad, angry, ...) **Classi**fication.

This repository is an attempt to tackle the task of predicting the facial expression,
age gender and race prediction from  facial images.

As I was planning to train the models on Google Colab platform, I decided to gather 
all my utility functions/classes in a package I named democlassi which is pip 
installable via `pip install --upgrade democlassi`.

The Training is done using Pytorch and pytorch-ignite and more 
information about the data used, the training process and live prediction using 
pretrained models and opencv can be found on 
[this github page](https://alkasaliss.github.io/DEmoClassi/).
