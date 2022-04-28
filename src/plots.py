import matplotlib.pyplot as plt

def plot_sample(batch, title=None):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    data = batch[0]
    im = plt.imshow(data*3600*24,
                    cmap='YlGnBu',
                    vmin=0,
                    vmax=15,
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