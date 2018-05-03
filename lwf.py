import experiment
import plotter
import json
import numpy as np
import os
folders = ['plots/scaling/']
experimentCon = []
for folder in folders:
    files = os.listdir(folder)
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
legend = ["Classifier with Scaling", "Classifier", "Ideal NMC"]
for e in experimentCon:
    ncm = []
    tc = []
    x = []
    ic = []
    for ex in e:
        x = ex.results['Trained Classifier'][0]
        ncm.append(ex.results['Trained Classifier Scaled'][1])
        tc.append(ex.results['Trained Classifier'][1])
        ic.append(ex.results['Ideal NMC'][1])
    ncmStd = np.std(ncm, axis=0)
    tcStd = np.std(tc, axis=0)
    ncm = np.mean(ncm,axis=0)
    tc = np.mean(tc, axis=0)
    print (ncmStd, tcStd)
    myPlotter.plot(x, ncm, legend=legend[counter]+" with NCM", error=ncmStd)
    myPlotter.plot(x, tc, legend=legend[counter]+" with TC", error=tcStd)
    myPlotter.plot(x, ic, legend=legend[counter] + " with TC", error=tcStd)
    counter+=1
myPlotter.save_fig("../tempLWF", 11)

