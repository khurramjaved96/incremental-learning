import experiment
import plotter
import json
import numpy as np
import os
folders = ['plots/herding2/herding/' , 'plots/herding2/noherding/']
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
legend = ["iCaRL with Herding", "iCaRL no Herding"]
counter= 0
for e in experimentCon:
    counter+=1
    ncm = []
    tc = []
    x = []
    ic = []
    inmc = []
    for ex in e:
        x = ex.results['Trained Classifier'][0]
        ic.append(ex.results['Trained Classifier'][1])
        inmc.append(ex.results['NMC'][1])
        ncm.append(ex.results['Ideal NMC'][1])
    ncmStd = np.std(ncm, axis=0)

    iinmc = np.std(inmc, axis=0)
    incm = np.std(ncm, axis=0)
    iic = np.std(ic, axis=0)
    inmc = np.mean(inmc, axis=0)
    ncm = np.mean(ncm, axis=0)
    ic = np.mean(ic,axis=0)
    if counter==1:
        myPlotter.plot(x, inmc, legend="iCaRL", error=iinmc)
    else:
        myPlotter.plot(x, inmc, legend="iCaRL without Herding", error=iinmc)
    # myPlotter.plot(x, ncm, legend="NCM (Ideal)"+str(counter), error=iinmc)
    # myPlotter.plot(x, ic, legend="Trained Classifier"+str(counter), error=iic)


    counter+=1

myPlotter.save_fig("../herding2", 101, yStart=0, xRange=0)

