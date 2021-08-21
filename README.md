# Ablation Studies on "SimCLR - A Simple Framework for Contrastive Learning of Visual Representations"

This repository is an extended version of SimCLRv1 from google-research. For detailed information on the original code please refer to the following repository: https://github.com/google-research/simclr.

It allows conducting ablation studies with respect to different similarity metrics (euclidean, mahalanobis distance) and image augmentation techniques (salt-and-pepper noise, brightness, inversion, per-image standardization).

## Enviroment setup

All my models were trained on a *single* GPU using Google Colab.

The code is compatible with both TensorFlow v1 and v2. See requirements.txt for all prerequisites, and you can also install them using the following command.

```
pip install -r requirements.txt
```


## Pretraining

To pretrain the original model on CIFAR-10, try the following command:

```
python run.py --train_mode=pretrain \
  --train_batch_size=256 --train_epochs=200 \
  --learning_rate=0.1 --weight_decay=1e-6 --temperature=0.5 \
  --dataset=cifar10 --image_size=32 --eval_split=test --resnet_depth=18 \
  --color_jitter_strength=0.5 
  --model_dir=/tmp/simclr_test --use_tpu=False
```

To pretrain the model using a different similarity metric (e.g. euclidean distance), change *similarity_measure* (and *hidden_norm*):

```
python run.py --train_mode=pretrain \
  --train_batch_size=256 --train_epochs=200 \
  --learning_rate=0.1 --weight_decay=1e-6 --temperature=0.5 \
  --dataset=cifar10 --image_size=32 --eval_split=test --resnet_depth=18 \
  --color_jitter_strength=0.5 
  --model_dir=/tmp/simclr_test --use_tpu=False\
  --similarity_measure=euclidean --hidden_norm=False \
```

To pretrain the model using different (combinations of) data augmentation techniques (e.g. only color distortion), try the following command:

```
python run.py --train_mode=pretrain \
  --train_batch_size=256 --train_epochs=200 \
  --learning_rate=0.1 --weight_decay=1e-6 --temperature=0.5 \
  --dataset=cifar10 --image_size=32 --eval_split=test --resnet_depth=18 \
  --model_dir=/tmp/simclr_test --use_tpu=False\
  --use_color_distort=True --color_jitter_strength=0.5 --use_crop=False --use_flip=False --use_blur=False \
  --use_salt_and_pepper=False --use_brightness=False --use_random_invert=False --use_standardize=False \
```


## Finetuning

To fine-tune a linear head (in the same manner as in the original paper), try the following command:

```
python run.py --mode=train_then_eval --train_mode=finetune \
  --fine_tune_after_block=4 --zero_init_logits_layer=True \
  --variable_schema='(?!global_step|(?:.*/|^)LARSOptimizer|head)' \
  --global_bn=False --optimizer=momentum --learning_rate=0.05 --weight_decay=0 \
  --train_epochs=20 --train_batch_size=128 --warmup_epochs=0 \
  --dataset=cifar10 --image_size=32 --eval_split=test --resnet_depth=18 \
  --checkpoint=/tmp/simclr_test \
  --model_dir=/tmp/simclr_test_ft --use_tpu=False \
  --similarity_measure=cosine --hidden_norm=True \
  --use_color_distort=False --use_crop=True --use_flip=True --use_blur=False \
  --use_salt_and_pepper=False --use_brightness=False --use_random_invert=False --use_standardize=False \
```

You can check the results using tensorboard, such as

```
python -m tensorboard.main --logdir=/tmp/simclr_test
```

As a reference, the above runs on CIFAR-10 using the original settings should give you around 71% accuracy.


## Disclaimer
This is not an official Google product.
