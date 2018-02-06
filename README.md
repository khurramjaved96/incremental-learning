# iCarl2.0
This is an on-going pytorch implementation of iCarl[1].
## Implemented Modules
1. Data pipeline
2. Data loader with upsampling
3. Resnet32, Densenet, and a simple network (The simple network is for debugging on a CPU based system).
4. Distillation loss.

## Remaining modules
1. Exemplars selection based on Nearest Mean Classifier criteria (Currently, randomly selected expemplars).
2. Nearest Mean Classification (Currently using the trained classifier. This is almost finished).

Note : We are reimplementing the paper even though the Theano code is already available because reimplementing clarifies a lot of confusions. Also, because we are iterating very quickly, there is little focus on documentation and comments. We will be adding those later.g

## References
[1] https://arxiv.org/abs/1611.07725