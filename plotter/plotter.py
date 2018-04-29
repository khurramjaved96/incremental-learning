import matplotlib.pyplot as plt
import matplotlib
import numpy as np

plt.switch_backend('agg')

MEDIUM_SIZE = 16

font = {'family' : 'sans-serif',
        'weight':'bold'}

matplotlib.rc('xtick', labelsize=MEDIUM_SIZE)
matplotlib.rc('ytick', labelsize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels

# matplotlib.rc('font', **font)
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


class Plotter():
    def  __init__(self):
        import itertools
        # plt.figure(figsize=(12, 9))
        self.marker = itertools.cycle(('o', '+', "v", "^", "8",'.', '*'))
        self.handles=[]
        self.lines = itertools.cycle(('--', '-.', '-', ':'))
    def plot(self,x,y, xLabel="Number of Classes",yLabel = "Accuracy %", legend="none",title=None, error=None):
        self.x = x
        self.y = y
        plt.grid(color='0.89', linestyle='--', linewidth=1.0)
        if error is None:
            l, = plt.plot(x,y,linestyle=next(self.lines), marker=next(self.marker), label=legend, linewidth=2)
        else:
            l = plt.errorbar(x, y, yerr=error, capsize=4.0, capthick=2.0, linestyle=next(self.lines), marker=next(self.marker), label=legend, linewidth=2.0)

        self.handles.append(l)
        self.x_label = xLabel
        self.y_label = yLabel
        if title is not None:
            plt.title(title)

    def save_fig(self, path, xticks=105):
        plt.legend(handles=self.handles)
        plt.ylim( (0, 100+1.2) )
        plt.xlim((0,xticks+.2))
        plt.ylabel(self.y_label)
        plt.xlabel(self.x_label)
        plt.yticks(list(range(10,105,10)))
        plt.xticks(list(range(0, xticks+1, int(xticks/10))))
        plt.savefig(path+".jpg")
        plt.savefig(path+".svg", format='svg', dpi=1200)
        plt.gcf().clear()

    def save_fig2(self, path, xticks=105):
        plt.legend(handles=self.handles)
        plt.xlabel("Memory Budget")
        plt.ylabel("Average Incremental Accuracy")
        plt.savefig(path+".jpg")
        plt.gcf().clear()

    def plotMatrix(self, epoch, path, img):
        import matplotlib.pyplot as plt
        plt.imshow(img, cmap='jet', interpolation='nearest')
        plt.colorbar()
        plt.savefig(path + str(epoch) + ".svg", format='svg', dpi=1200)
        plt.gcf().clear()

    def saveImage(self, img, path, epoch):
        from PIL import Image
        im = Image.fromarray(img)
        im.save(path + str(epoch) + ".jpg")

    def plot_histogram(self, subplots, embeddings, range_val, colors):
        p1, p2, p3 = subplots
        bins = np.linspace(range_val[0], range_val[1], 100)
        p1.hist(embeddings[0], bins, alpha=0.5, edgecolor="k", color=colors[0])
        p2.hist(embeddings[1], bins, alpha=0.5, edgecolor="k", color=colors[1])
        p3.hist(embeddings[0] - embeddings[1], bins, alpha=0.5, edgecolor="k", color=colors[2])
        plt.xlabel('Value')
        plt.ylabel('Frequency')

    def plot_chart(self, subplots, embeddings, range_val, colors):
        p1, p2, p3 = subplots
        p1.plot(embeddings[0], color=colors[0])
        p2.plot(embeddings[1], color=colors[1])
        p3.plot(np.abs(embeddings[0] - embeddings[1]), color=colors[2])
        plt.xlabel('Logit')
        plt.ylabel('Value')

    def save_embedding_plot(self, subplots, labels, plot_name, experiment):
        p1, p2, p3 = subplots
        p1.legend([labels[0]], loc="upper right")
        p2.legend([labels[1]], loc="upper right")
        p3.legend(["Difference"], loc="upper right")
        plt.savefig(experiment.path + "embedding_" + plot_name + ".jpg")
        plt.gcf().clear()

    def plot_embeddings(self, embeddings, labels, range_val, experiment, plot_name=""):
        """
        Plots Histogram and Chart of two embeddings and their difference
        Parameters:
            embeddings: Array containing 2x 1D vector of embeddings
            labels:     Array containing labels of the corresponding embeddings
            range_val:  Range of logit values for the histogram
        """
        colors = ['b','g', 'r']
        #Plot histogram
        f, (subplots) = plt.subplots(3, 1, sharex=True, sharey=True)
        self.plot_histogram(subplots, embeddings, range_val, colors)
        self.save_embedding_plot(subplots, labels, "hist_" + plot_name, experiment)
        #Plot chart
        f, (subplots) = plt.subplots(3, 1, sharex=True, sharey=True)
        self.plot_chart(subplots, embeddings, range_val, colors)
        self.save_embedding_plot(subplots, labels, "chart_" + plot_name, experiment)

if __name__=="__main__":
   import json
   from pprint import pprint
   import numpy as np

   pl = Plotter()
   x = [2,4,6,8,10]
   #MNIST 20 epoch acdistillation
   #icarl_nmc = [99.89832231825115, 98.26574853353736, 97.56890943331621, 96.8427753023552, 97.11]
   #icarl_trained = [99.89832231825115, 97.06707472583524, 93.51138503680876, 88.24952259707193, 81.07]
   #lwf = [99.94916115912557, 92.39989798520786, 81.7668207498716, 73.17632081476766, 66.89]
   #gan_trained = [99.55044955044956, 95.46502690238279, 95.67003251754235, 94.61489497135582, 94.35]
   #gan_nmc = [99.55044955044956, 95.23443504996156, 95.61868902960808, 94.62762571610439, 94.48]
   #gan_ideal = [99.55044955044956, 96.38739431206764, 96.04655142906041, 95.00954805856142, 94.78]
   #forgetting = [99.94916115912557, 49.73221117061974, 32.83684300633453, 25.626989178866964, 21.45]
   icarl_nmc = [98.6, 93.5, 89.66666666666667, 83.75, 77.95]
   icarl_trained = [98.85, 88.7, 81.53333333333333, 70.4125, 62.01]
   lwf = [98.2, 75.725, 59.78333333333333, 48.625, 31.33]
   gan_trained = [96.7, 81.2, 67.03333333333333, 54.0, 41.11]
   gan_nmc = [96.6, 83.05, 72.8, 58.8875, 48.84]
   gan_ideal = [96.65, 82.925, 72.55, 59.95, 48.48]
   forgetting = [98.4, 47.875, 31.8, 24.075, 19.77]

   legends = ["iCarl NCM", "iCarl Trained Classifier", "LwF", "GAN Trained Classifier", "GAN NCM", "GAN Ideal NCM", "Catastrophic Forgetting"]

   pl.plot(x, icarl_nmc, legend=legends[0])
   pl.plot(x, gan_nmc, legend=legends[4])
   pl.plot(x, icarl_trained, legend=legends[1])
   pl.plot(x, gan_trained, legend=legends[3])
   #pl.plot(x, gan_ideal, legend=legends[5])
   pl.plot(x, lwf, legend=legends[2])
   pl.plot(x, forgetting, legend=legends[6])
   pl.save_fig("CIFAR10", 10)
