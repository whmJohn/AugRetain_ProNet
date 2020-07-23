# AugRetain_ProNet

## Requirements

Python 3.7

Pytorch 1.3

## Data

For Omniglot experiments, I directly attach omniglot 28x28 resized images in the git, which is created based on [omniglot](https://github.com/brendenlake/omniglot) and [maml](https://github.com/cbfinn/maml).

For mini-Imagenet experiments, please download [mini-Imagenet](https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE) and put it in ./datas/mini-Imagenet and run proc_image.py to preprocess generate train/val/test datasets. (This process method is based on [maml](https://github.com/cbfinn/maml)).

## Train

For now, I completed the omniglot 5-way 1-shot and 5-way 5-shot part, you can train your own model by changing the model save part.

omniglot 5-way 1-shot:

```
python omniglot_train_one_shot.py -w 5 -s 1 -b 19 
```

omniglot 5-way 5-shot:

```
python omniglot_train_few_shot.py -w 5 -s 5 -b 15 
```

mini-Imagenet 5-way 1-shot:

```
python miniimagenet_train_one_shot.py -w 5 -s 1 -b 15
```

mini-Imagenet 5-way 5-shot:

```
python miniimagenet_train_few_shot.py -w 5 -s 5 -b 10
```

you can change -b parameter based on your GPU memory. Currently It will load my trained model, if you want to train from scratch, you can delete models by yourself.

## Test

omniglot 5-way 1-shot:

```
python omniglot_test_one_shot.py -w 5 -s 1
```

omniglot 5-way 5-shot:

```
python omniglot_test_few_shot.py -w 5 -s 5  
```

