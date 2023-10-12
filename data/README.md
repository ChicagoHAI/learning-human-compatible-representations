# Image Data and Triplet Annotations

This directory contains the image data and triplet annotations for the
Butterflies v.s. Moths dataset and the Chest X-rays dataset used in our paper.

## Butterfly v.s. Moths Dataset

The butterfly v.s. moths dataset is a subset of the ImageNet dataset. It 
contains 200 images of four species of butterflies and moths.

We recruit 80 participants from Prolific to annotate the triplets. Each
participant is asked to compare 50 triplets of images and select the image
that is most similar to the anchor image.
There are in total 4000 triplet annotations.

## Chest X-rays Dataset

The chest X-rays dataset is a subset of the Chest X-ray dataset from 
[Kermany et al. (2018)](https://www.sciencedirect.com/science/article/pii/S0092867418301545).
It contains 3166 images of chest X-ray images. We created a balanced subset
with either a normal or pneumonia label. We provide the filenames of our
splits in `pneumonia.txt`. The corresponding images can be downloaded from 
[the url to the dataset in the original paper](https://data.mendeley.com/datasets/rscbjbr9sj/2).

We recruit 100 crowdworkers from Prolific to annotate the triplets. Each
crowdworker is asked to answer 20 triplet questions and select the image
that is most similar to the anchor image.
There are in total 2000 triplet annotations.
