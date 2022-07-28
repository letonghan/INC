#!/bin/sh
set -x

echo "Installing requirements"
pip install -r requirements.txt
pip install neural_compressor
cd examples/pytorch/image_recognition/MobileNetV2-0.35/distillation/eager
pip install -r requirements.txt
echo "======= pip list ======="
pip list
echo "Training teacher model WideResNet40-2"
python -u train_without_distillation.py --epochs 1 --lr 0.1 --layers 40 --widen-factor 2 --name WideResNet-40-2 --tensorboard
echo "Distillation"
python -u main.py --epochs 1 --lr 0.02 --name MobileNetV2-0.35-distillation --teacher_model runs/WideResNet-40-2/model_best.pth.tar --tensorboard --seed 9

