import matplotlib.pyplot as plt
plt.switch_backend('agg')

class Plotter():
    def  __init__(self):
        import itertools
        self.marker = itertools.cycle(('o', '+', '.', '*', '*'))
        self.handles=[]
    def plot(self,x,y, xLabel="Number of Classes",yLabel = "Accuracy", legend="none",title="none"):
        self.x = x
        self.y = y
        plt.grid(color='b', linestyle='--', linewidth=0.2)
        l, = plt.plot(x,y,linestyle='-', marker=next(self.marker), label=legend)

        self.handles.append(l)
        self.x_label = xLabel
        self.y_label = yLabel
        plt.title(title)

    def save_fig(self, path, xticks=105):
        plt.legend(handles=self.handles)
        plt.ylim( (0, 100) )
        plt.xlim((0,xticks))
        plt.ylabel(self.y_label)
        plt.xlabel(self.x_label)
        plt.yticks(list(range(0,105,10)))
        plt.xticks(list(range(0, xticks, int(xticks/10))))
        plt.savefig(path+"_globalPlot_.eps",format='eps', dpi=1200)

if __name__=="__main__":
    pl = Plotter()
    pl.plot([1,2,3,4], [2,3,6,2])
    pl.save_fig("test.jpg")
