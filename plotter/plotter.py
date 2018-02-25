import matplotlib.pyplot as plt
import matplotlib

plt.switch_backend('agg')


MEDIUM_SIZE = 14

font = {'family' : 'sans-serif'}

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels

# matplotlib.rc('font', **font)
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
# plt.style.use('ggplot')
print(plt.style.available)

class Plotter():
    def  __init__(self):
        import itertools
        # plt.figure(figsize=(12, 9))
        self.marker = itertools.cycle(('o', '+', "v", "^", "8",'.', '*'))
        self.handles=[]
        self.lines = itertools.cycle(('--', '-.', '-', ':'))
    def plot(self,x,y, xLabel="Number of Classes",yLabel = "Accuracy", legend="none",title=None, error=None):
        self.x = x
        self.y = y
        plt.grid(color='0.86', linestyle='--', linewidth=0.5)
        if error is None:
            l, = plt.plot(x,y,linestyle=next(self.lines), marker=next(self.marker), label=legend, linewidth=2.0)
        else:
            l = plt.errorbar(x, y, yerr=error, capsize=4.0, capthick=2.0, linestyle=next(self.lines), marker=next(self.marker), label=legend, linewidth=2.0)

        self.handles.append(l)
        self.x_label = xLabel
        self.y_label = yLabel
        if title is not None:
            plt.title(title)

    def save_fig(self, path, xticks=105):
        plt.legend(handles=self.handles)
        plt.ylim( (0, 100) )
        plt.xlim((0,xticks+.2))
        plt.ylabel(self.y_label)
        plt.xlabel(self.x_label)
        plt.yticks(list(range(0,105,10)))
        plt.xticks(list(range(0, xticks+1, int(xticks/10))))
        plt.savefig(path+"globalPlot_.eps",format='eps', dpi=1200)
        plt.gcf().clear()

if __name__=="__main__":
    pl = Plotter()
    pl.plot([1,2,3,4], [2,3,6,2])
    pl.save_fig("test.jpg")
