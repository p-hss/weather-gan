import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
from os.path import expanduser
cartopy.config['pre_existing_data_dir'] = expanduser('/p/tmp/hess/cartopy/shapefiles/natural_earth/physical/')

def plot_map(data, ax, vmin, vmax, cmap):

    lats = np.arange(-90,90,2.5)
    lons = np.arange(182,360,2.5)
    lons, lats = np.meshgrid(lons, lats)

    im = ax.pcolormesh(lons, lats, data,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax)
    ax.coastlines()
    gl = ax.gridlines(crs=ccrs.PlateCarree(),
                      draw_labels=True,
                      linewidth=1,
                      color='gray',
                      alpha=0.5,
                      linestyle='--')
    gl.right_labels = False
    gl.top_labels = False
    return im

def plot_sample(batch, title=None):
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    data = batch[0]
    im = plot_map(data*3600*24, ax, 0, 15, 'YlGnBu')
    plt.colorbar(im, label='Precipitation [mm/day]')
    plt.title(title)

    ax = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    data = batch[1]
    im = plot_map(data, ax, 230, 310, 'jet')
    plt.colorbar(im, label='Temperature [K]')
    plt.title(title)
    plt.show()


def plot_histograms(gan, target):
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    
    plt.hist(target[:,1].cpu().numpy().flatten(),
             histtype='step',
             density=True,
             bins=50,
             label='CMIP6',
             color='blue')
    
    plt.hist(gan[:,1].cpu().numpy().flatten(),
             histtype='step',
             density=True,
             bins=50,
             label='GAN',
             color='red')  
    
    plt.ylabel('Relative frequency')
    plt.xlabel('Temperature [K]')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.hist((target[:,0]*24*3600).cpu().numpy().flatten(), 
             histtype='step',
             density=True,
             log=True,
             bins=50,
             label='CMIP6',
             color='blue')
             
             
    
    plt.hist((gan[:,0]*24*3600).cpu().numpy().flatten(),
             histtype='step',
             density=True,
             log=True,
             bins=50,
             label='GAN',
             color='red')
    
    plt.ylabel('Relative frequency')
    plt.xlabel('Precipitation [mm/day]')
    plt.legend()
    plt.show()