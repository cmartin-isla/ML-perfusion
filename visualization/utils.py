import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

signature = np.random.rand(30,19)

def plot_3d_bar(signature):
    # setup the figure and axes
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(111, projection='3d')
    
    # fake data
    _x = np.arange(signature.shape[0])
    _y = np.arange(signature.shape[1])
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    
    
    top = signature.ravel()
    bottom = np.zeros_like(top)
    width = depth = 1
    
    colors = plt.cm.jet(signature.flatten()/float(signature.max()))
    
    ax1.bar3d(x, y, bottom, width, depth, top, shade=True, color = colors)
    ax1.set_title('Shaded')
    
    

    plt.show()


def plot_sequences(signature):


    for row in range(signature.shape[0]):
    
        plt.plot(signature[row,:],'-')
    
    plt.show()

