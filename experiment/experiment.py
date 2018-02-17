class experiment:
    '''
    Class to store results of any experiment 
    '''
    def __init__(self, args):
        self.name = None
        self.params = vars(args)
        self.args = args
        self.results= {}
        self.constructExperimentName(args)
    def constructExperimentName(self,args):
        import os
        name = [args.model_type, str(args.epochs_class), str(args.step_size)]
        if not args.no_herding:
            name.append("herding")
        if not args.no_distill:
            name.append("distillation")
        ver = 0
        while os.path.exists("../" + args.name+"_"+str(ver)):
            ver+=1

        os.makedirs("../" + args.name+"_"+str(ver))

        self.name = "_".join(name) +"_"+str(ver)
        self.path = "../" + args.name +"_"+str(ver)+ "/" + "_".join(name)

        return "../" + args.name +"_"+str(ver)+ "/" + "_".join(name)