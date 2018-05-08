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
            l, = plt.plot(x,y,linestyle=next(self.lines), marker=next(self.marker), label=legend, linewidth=2.0)
        else:
            l, = plt.plot(x,y,linestyle=next(self.lines), marker=next(self.marker), label=legend, linewidth=2.0)
            y, error = np.array(y), np.array(error)
            plt.fill_between(x, y-error, y+error, alpha=0.12)
            #l = plt.errorbar(x, y, yerr=error, capsize=4.0, capthick=2.0, linestyle=next(self.lines), marker=next(self.marker), label=legend, linewidth=2.0)

        self.handles.append(l)
        self.x_label = xLabel
        self.y_label = yLabel
        if title is not None:
            plt.title(title)

    def save_fig(self, path, xticks=105):
        plt.legend(handles=self.handles)
        plt.ylim((0, 100+1.2) )
        plt.xlim((0,xticks+.2))
        #plt.ylabel(self.y_label)
        plt.xlabel(self.x_label)
        plt.yticks(list(range(10,105,10)))
        plt.xticks(list(range(0, xticks+1, int(xticks/10))))
        plt.savefig(path+".jpg")
        plt.savefig(path+".pdf", format='pdf', dpi=600)
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
    pl = Plotter()
    pl.plot([1,2,3,4], [2,3,6,2])
    pl.save_fig("test.jpg")
