import experiment
import plotter
import json
import numpy as np
import os
folders = ['plots/memorybudget/us/']
experimentCon = []
for folder in folders:
    files = os.listdir(folder)
    files = np.sort([int(i) for i in files])
    files = [str(f) for f in files]
    experiments= []
    for f in files:
        print (f)
        if ".DS" in f:
            continue
        with open(folder+f) as json_data:
            d = json.load(json_data)
            tempDic = json.loads(d)
            e = experiment.experiment(None, None,None)
            e.__dict__ = tempDic
            experiments.append(e)
    experimentCon.append(experiments)

counter=0
myPlotter = plotter.Plotter()
legend = ["Uniform sampling"]
for e in experimentCon:
    ncm = []
    tc = []
    x = []
    for ex in e:
        print (ex.__dict__)
        x.append(ex.params['memory_budget'])
        ncm.append(np.mean(ex.results['NCM'][1]))
        tc.append(np.mean(ex.results['Trained Classifier'][1]))
    print (ncm, x, tc)
    # ncm = np.mean(ncm,axis=1)
    # tc = np.mean(tc, axis=1)
    myPlotter.plot(x, ncm, legend=legend[counter]+" with NCM")
    myPlotter.plot(x, tc, legend=legend[counter]+" with TC")
    counter+=1
myPlotter.save_fig2("../temp", 10)

