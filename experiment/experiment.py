import json
import os

class experiment:
    '''
    Class to store results of any experiment 
    '''
    def __init__(self, name, args, outputDir="../"):
        self.name = name
        self.params = vars(args)
        self.results = {}
        self.dir = outputDir

        ver = 0
        while os.path.exists("../" + self.name + "_" + str(ver)):
            ver += 1

        os.makedirs("../" + self.name + "_" + str(ver))
        self.path = "../" + self.name + "_" + str(ver) + "/" + name

        self.results["Temp Results"]= [[1,2,3,4], [5,6,2,6]]

    def storeJSON(self):
        with open(self.path +"JSONDump", 'w') as outfile:
            json.dump(json.dumps(self.__dict__), outfile)


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='iCarl2.0')
    args = parser.parse_args()
    e = experiment("TestExperiment", args)
    e.storeJSON()

