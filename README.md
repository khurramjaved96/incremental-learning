# iCarl2.0
This is an on-going pytorch implementation of iCarl[1].

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
## Implemented Modules
1. Data pipeline
2. Data loader with upsampling
3. Resnet32, Densenet, and a simple network (The simple network is for debugging on a CPU based system).
4. Distillation loss.

## Ideas to implement and test
1. Auto-encoder to preserve information of a class 
2. Weights for binary cross entropy for significantly faster training

Note : We are reimplementing the paper even though the Theano code is already available because reimplementing clarifies a lot of confusions. Also, because we are iterating very quickly, there is little focus on documentation and comments. We will be adding those later.

## References
[1] https://arxiv.org/abs/1611.07725
