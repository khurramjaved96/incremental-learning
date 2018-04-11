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
                        [--depth DEPTH] [--Dataset DATASET]
```
## Dependencies 

1. Pytorch 0.3.0.post4
2. Python 3.6 
3. torchnet (https://github.com/pytorch/tnt) 
4. OpenCV 

## Install Instructions 
1. Install Anaconda with Python 3
2. Install pytorch and torchnet 
3. Install OpenCV from conda repository
Done. 

## Branches
1. GAN driven incremental learning is being done in the branch gan.
2. Auto-encoder based knowledge preservation is being implemented in branch autoencoders

=======
## Results 
![alt text](http://khurramjaved96.github.io/random/result.jpg "Incremental Learning on MNIST")


## References
[1] https://arxiv.org/abs/1611.07725
