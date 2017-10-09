import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from IPython.display import HTML
from matplotlib import rc


def imshow(img, fig=None, **kwargs):
    img = np.squeeze(img)
    fig = fig if fig is not None else plt.gcf()
    ax = fig.add_axes([0,0,1,1])
    ax.axis('off')
    return ax.imshow(img, **kwargs)


def generate_animation (sampler, zs, steps=100, shape=(200,200), duration=5):
    rc('animation', html='html5')

    x_dim, y_dim = shape
    images       = []

    for zn1, zn2 in zip(zs, zs[1:]):
        images += sampler.morph(zn1
                               , zn2
                               , n_total_frame=steps
                               , x_dim=x_dim
                               , y_dim=y_dim
                               )
    frames = len(images)

    fig = plt.figure(0, figsize=(5, 5))
    img_ax = imshow(images[0], animated=True, fig=fig, cmap='Greys')

    def updatefig(n):
        img_ax.set_array(np.squeeze(images[n]))
        return img_ax,

    plt.close(fig)
    interval = duration * 1000 / steps

    return animation.FuncAnimation(fig
                                  , updatefig
                                  , frames=frames
                                  , interval=interval
                                  , blit=True) 
