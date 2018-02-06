import matplotlib.pyplot as plt
plt.switch_backend('agg')

class plotter():
    def  __init__(self):
        pass
    def plot(self,x,y, xLabel="Number of Classes",yLabel = "Accuracy"):
        self.x = x
        self.y = y
        self.xLabel = xLabel
        self.yLabel = yLabel
    def saveFig(self, path):
        plt.plot(self.x, self.y)
        plt.ylim( (0, 100) )
        plt.xlim((0,100))
        plt.ylabel(self.yLabel)
        plt.xlabel(self.xLabel)
        plt.savefig(path)

if __name__=="__main__":
    pl = plotter()
    pl.plot([1,2,3,4], [2,3,6,2])
    pl.saveFig("test.jpg")
