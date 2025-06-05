# Backdoor Attack with FDFT

This is the official implementation of our paper "An Invisible Backdoor Attack through Adding Fixed Patch Triggers in Frequency Domain". This research project is developed based on Python 3 and Pytorch.

## Requirements

We have tested the code under the following environment settings:

- python = 3.8.20
- pytorch = 1.13.1
- torchvision = 0.14.1

## Quick Start

**Step 1: Train Clean model**

In FDFT, we train a clean model.

```
python train_clean_cifar.py --model resnet18 --save_surrogate save_surrogate --epochs 100 --device 1
```

**Step 2: Train backdoored model**

With the fixed patch trigger, we train the backdoored model.

```
python train_poison_cifar.py --save_dir save_backdoor --y_target 0 --epochs 100 --poison_rate 0.01 --device 1
```
