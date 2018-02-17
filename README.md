# iCarl2.0
This is an on-going pytorch implementation of iCarl[1].

## Interface to run experiments

``` bash
runExperiment.py [-h] [--batch-size N] [--test-batch-size N]
                        [--epochs N] [--lr LR]
                        [--schedule SCHEDULE [SCHEDULE ...]]
                        [--gammas GAMMAS [GAMMAS ...]] [--momentum M]
                        [--no-cuda] [--no-distill] [--no-random]
                        [--no-herding] [--oversampling] [--seed S]
                        [--log-interval N] [--model-type MODEL_TYPE]
                        [--name NAME] [--sortby SORTBY] [--decay DECAY]
                        [--step-size STEP_SIZE]
                        [--memory-budget MEMORY_BUDGET]
                        [--epochs-class EPOCHS_CLASS] [--classes CLASSES]
                        [--depth DEPTH] [--dataset DATASET]
```
## Dependencies 

1. Pytorch 0.3.0.post4
2. Python 3.6 
3. torchnet (https://github.com/pytorch/tnt) 

<<<<<<< HEAD
## Ideas to implement and test
1. Auto-encoder to preserve information of a class 
2. Generative models for data seen in the past to remove dependency on examplers.
=======
## Results 
![alt text](http://khurramjaved96.github.io/random/result.jpg "Incremental Learning on MNIST")
>>>>>>> acd7acec3baefd5ec7bf83c61e705acf4387d7a1


## References
[1] https://arxiv.org/abs/1611.07725
