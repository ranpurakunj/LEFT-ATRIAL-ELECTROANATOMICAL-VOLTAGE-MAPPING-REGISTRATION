#! /usr/bin/bash

# for output
mkdir -p ${HOME}/results

# Python3 command
PY3="nice -n 10 python"

# first, traing a classifier.
# ModelNet categories are given in './sampledata/modelnet40_half1.txt' (as examaple)
${PY3} drive/MyDrive/PointNetLK-master/PointNetLK-master/experiments/train_classifier.py \
 -o ${HOME}/results/ex1_classifier_0915 \
 -i /content/drive/MyDrive/ml/ModelNet40 \
 -c drive/MyDrive/PointNetLK-master/PointNetLK-master/experiments/sampledata/modelnet40_half1.txt \
 -l ${HOME}/results/ex1_classifier_0915.log \
 --device cuda:2

# the one of the results is '${HOME}/results/ex1_classifier_0915_feat_best.pth'
# this file is the model that computes PointNet feature.

# train PointNet-LK. fine-tune the PointNet feature for classification (the above file).
${PY3} drive/MyDrive/PointNetLK-master/PointNetLK-master/experiments/train_pointlk.py \
 -o ${HOME}/results/ex1_pointlk_0915 \
 -i /content/drive/MyDrive/ml/ModelNet40 \
 -c drive/MyDrive/PointNetLK-master/PointNetLK-master/experiments/sampledata/modelnet40_half1.txt \
 -l ${HOME}/results/ex1_pointlk_0915.log \
 --transfer-from ${HOME}/results/ex1_classifier_0915_feat_best.pth \
 --epochs 400 \
 --device cuda:2

# the trained model:
# ${HOME}/results/ex1_pointlk_0915_model_best.pth

#EOF
