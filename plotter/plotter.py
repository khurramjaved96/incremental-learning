import matplotlib.pyplot as plt
import matplotlib
import numpy as np

plt.switch_backend('agg')

class Plotter():
    def  __init__(self):
        import itertools
        self.marker = itertools.cycle(('o', '+', '.', '*', '*'))
        self.handles=[]
    def plot(self,x,y, x_label="Number of Classes",y_label = "Accuracy", legend="none",title="none"):
        self.x = x
        self.y = y
        plt.grid(color='b', linestyle='--', linewidth=0.2)
        l, = plt.plot(x,y,linestyle='-', marker=next(self.marker), label=legend)

        self.handles.append(l)
        self.x_label = x_label
        self.y_label = y_label
        plt.title(title)

    def save_fig(self, path, xticks=105):
        plt.legend(handles=self.handles)
        plt.ylim( (0, 100) )
        plt.xlim((0,xticks))
        plt.ylabel(self.y_label)
        plt.xlabel(self.x_label)
        plt.yticks(list(range(0,105,10)))
        plt.xticks(list(range(0, xticks, int(xticks/10))))
        plt.savefig(path)
        plt.gcf().clear()

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
