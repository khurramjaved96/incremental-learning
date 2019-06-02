# Revisiting Distillation and Incremental Classifier Learning
Accepted at ACCV18. Pre-print is available at : http://arxiv.org/abs/1807.02802

Citing the paper : 
``` bash 
@InProceedings{10.1007/978-3-030-20876-9_1,
author="Javed, Khurram
and Shafait, Faisal",
editor="Jawahar, C.V.
and Li, Hongdong
and Mori, Greg
and Schindler, Konrad",
title="Revisiting Distillation and Incremental Classifier Learning",
booktitle="Computer Vision -- ACCV 2018",
year="2019",
publisher="Springer International Publishing",
address="Cham",
pages="3--17",
abstract="One of the key differences between the learning mechanism of humans and Artificial Neural Networks (ANNs) is the ability of humans to learn one task at a time. ANNs, on the other hand, can only learn multiple tasks simultaneously. Any attempts at learning new tasks incrementally cause them to completely forget about previous tasks. This lack of ability to learn incrementally, called Catastrophic Forgetting, is considered a major hurdle in building a true AI system.",
isbn="978-3-030-20876-9"
}
``` 
## Interface to Run Experiments

``` bash
usage: runExperiment.py [-h] [--batch-size N] [--lr LR]
                        [--schedule SCHEDULE [SCHEDULE ...]]
                        [--gammas GAMMAS [GAMMAS ...]] [--momentum M]
                        [--no-cuda] [--random-init] [--no-distill]
                        [--distill-only-exemplars] [--no-random]
                        [--no-herding] [--seeds SEEDS [SEEDS ...]]
                        [--log-interval N] [--model-type MODEL_TYPE]
                        [--name NAME] [--outputDir OUTPUTDIR] [--upsampling]
                        [--pp] [--distill-step] [--hs]
                        [--unstructured-size UNSTRUCTURED_SIZE]
                        [--alphas ALPHAS [ALPHAS ...]] [--decay DECAY]
                        [--alpha-increment ALPHA_INCREMENT] [--l1 L1]
                        [--step-size STEP_SIZE] [--T T]
                        [--memory-budgets MEMORY_BUDGETS [MEMORY_BUDGETS ...]]
                        [--epochs-class EPOCHS_CLASS] [--dataset DATASET]
                        [--lwf] [--no-nl] [--rand] [--adversarial]
```

Default configurations can be used to run with same parameters as used by iCaRL. 
Simply run:
``` bash
python run_experiment.py
```
## Dependencies 

1. Pytorch 0.3.0.post4
2. Python 3.6 
3. torchnet (https://github.com/pytorch/tnt) 
4. tqdm (pip install tqdm)

Please see requirements.txt for a complete list. 

## Setting up enviroment 
The easiest way to install the required dependencies is to use conda package manager. 
1. Install Anaconda with Python 3
2. Install pytorch and torchnet 
3. Install tqdm (pip install progressbar2)
Done. 

## Branches
1. iCaRL + Dynamic Threshold Moving is implemented in "Autoencoders" branch.

=======
## Selected Results 
### Removing Bias by Dynamic Threshold Moving
![alt text](https://github.com/Khurramjaved96/incremental-learning/blob/autoencoders/images/thresholdmoving.png "Dynamic Threshold Moving on MNIST")
Result of threshold moving with T = 2 and 5. Note that different scale is used for
the y axis, and using higher temperature in general results in less bias.

### Confusion Matrix with and without Dynamic Threshold Moving
![alt text](https://github.com/Khurramjaved96/incremental-learning/blob/autoencoders/images/confusion.png "Dynamic Threshold Moving Confusion Matrix")
Confusion matrix of results of the classifier with (right) and without (left) threshold
moving with T=2. We removed the first five classes of MNIST from the train set and only
distilled the knowledge of these classes using a network trained on all classes. Without
threshold moving the model struggled on the older classes. With threshold moving, however,
not only was it able to classify unseen classes nearly perfectly, but also its performance did
not deteriorate on new classes
### Experiment Meta-file Details
![alt text](https://github.com/Khurramjaved96/incremental-learning/blob/autoencoders/images/protocol.png "Dynamic Threshold Moving Confusion Matrix")
Protocol used to store the state of an experiment. The green coded text is the git hash corresponding to the version of the repository used to run the experiment, the blue coded string is the arguments used for running the experiment, and the red coded string has the results of the experiment. By storing all three, we are able to easily reproduce the results and compare to existing results

## FAQs
### How do I implement more models? 
A. Add the model in model/ModelFactory and make sure the forward method of the model satisfy the API of model/resnet32.py
### How do I add a new dataset? 
A. Add the new dataset in DatasetFactory and specify the details in the dataHandler/dataset.py class. Make sure the dataset implements all the variables set by other datasets. 

## References
[1] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015

[2] Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H Lampert. Icarl: Incremental classifier and representation learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2001–2010, 2017.

[3] Zhizhong Li and Derek Hoiem. Learning without forgetting. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017.
