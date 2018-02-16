class experiment:
    '''
    Class to store results of any experiment 
    '''
    def __init__(self, params):
        self.name = None
        self.params = params
        self.results= {}

    def constructExperimentName(self, args):
        import os
        name = [args.model_type, str(args.epochs_class), str(args.step_size)]
        if not args.no_herding:
            name.append("herding")
        if not args.no_distill:
            name.append("distillation")
        ver = 0
        while not os.path.exists("../" + args.name+str(ver)):
            os.makedirs("../" + args.name+str(ver))

        self.name = "_".join(name) + str(ver)
        self.path = "../" + args.name +str(ver)+ "/" + "_".join(name)

        return "../" + args.name +str(ver)+ "/" + "_".join(name)