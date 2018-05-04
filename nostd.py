import experiment
import plotter
import json
import numpy as np
import os
folders = ['plots/herding/']
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
legend = ["iCaRL with Herding", "iCaRL without Herding"]
for e in experimentCon:
    ncm = []
    tc = []
    x = []
    ic = []
    for ex in e:
        x = ex.results['Trained Classifier'][0]
        ncm = ex.results['NMC'][1]
        myPlotter.plot(x, ncm, legend=legend[counter])
        counter+=1
myPlotter.save_fig("../tempLWF", 103, title="Herding vs No Herding")

