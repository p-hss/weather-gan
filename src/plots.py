import matplotlib.pyplot as plt

def plot_sample(batch):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    target = batch[0]['target'][0][0].cpu()
    im = plt.imshow(target*3600*24, vmax=20, cmap='YlGnBu', origin='lower')
    plt.colorbar(im, label='Precipitation [mm/day]')

    plt.subplot(1,2,2)
    target = batch[0]['target'][0][1].cpu()
    im = plt.imshow(target, cmap='seismic', origin='lower')
    plt.colorbar(im, label='Temperature [K]')
    plt.show()