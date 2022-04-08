import matplotlib.pyplot as plt

def plot_sample(batch, title=None):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    data = batch[0]
    im = plt.imshow(data*3600*24,
                    cmap='YlGnBu',
                    vmin=0,
                    vmax=20,
                    origin='lower')
    plt.colorbar(im, label='Precipitation [mm/day]')
    plt.title(title)

    plt.subplot(1,2,2)
    data = batch[1]
    im = plt.imshow(data,
                    cmap='seismic',
                    vmin=230,
                    vmax=310,
                    origin='lower')
    plt.colorbar(im, label='Temperature [K]')
    plt.title(title)
    plt.show()