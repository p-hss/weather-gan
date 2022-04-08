import os
import torch
import xarray as xr
import numpy as np
#from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm

from src.model import WeatherGenerator
from src.utils import config_from_file
from src.data import DataModule, Transforms
from src.plots import plot_sample


class Inference():

    """ Execute model on test data and return output as NetCDF. """
    
    def __init__(self,
                 checkpoint_path,
                 max_num_inference_steps=None,
                 epoch_index=None
                 ):
        
        self.checkpoint_path = checkpoint_path
        self.config_path = '/home/ftei-dsw/data/weather-gan/config-files/'
        self.config = self.load_config()
        self.replace_missing_configuration()
        self.results_path = self.config.results_path

        self.train_start = str(self.config.train_start)
        self.train_end = str(self.config.train_end)
        self.test_start = str(self.config.test_start)
        self.test_end = str(self.config.test_end)

        self.model = None
        self.generator = None
        self.discriminator = None
        self.model_output = None
        self.dataset = None
        self.epoch_index = epoch_index
        self.transforms = Transforms()


        self.max_num_inference_steps = max_num_inference_steps
        self.tst_batch_sz = 64
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    def load_config(self):
        self.uuid = self.get_uuid_from_path(self.checkpoint_path)
        config = config_from_file(f'{self.config_path}config_model_{self.uuid}.json')
        return config


    def replace_missing_configuration(self):
        """ adding parameters that have been added to config later,
            to ensure compatibility.
        """
        if hasattr(self.config, 'n_critic') is False:
            self.config.n_critic = 5 
        if hasattr(self.config, 'discriminator_num_layers') is False:
            self.config.discriminator_num_layers = 3 


    def run(self):
        
        files = self.get_files(self.checkpoint_path)

        if self.epoch_index is not None:
            files = [files[self.epoch_index-1]]

        for i, fname in enumerate(files):
            self.checkpoint_idx = i+1
            self.num_checkpoints = len(files)
            print(f'Checkpoint {self.checkpoint_idx} / {self.num_checkpoints}:')
            print(fname)
            print('')

            self.load_model(fname)
            self.execute_inference()
            self.apply_inverse_transforms()
            
        return self.model_output


    def get_files(self, path: str):
        if os.path.isfile(path):
            files = []
            files.append(path) 
        else:
            files = os.listdir(path)
            for i, f in enumerate(files):
                files[i] = os.path.join(path, f) 
        return files 


    def get_uuid_from_path(self, path: str):
        import re
        uuid4hex = re.compile('[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}', re.I)
        uuid = uuid4hex.search(path).group(0)
        return uuid

        
    def load_model(self, path):
        model = WeatherGenerator(self.config).load_from_checkpoint(checkpoint_path=path)
        model.freeze()
        generator = model.generator
        self.generator = generator.to(self.device)


    def get_dataloader(self):
        datamodule = DataModule(self.config)
        datamodule.setup("test")
        dataloader = datamodule.test_dataloader()
        return dataloader


    def execute_inference(self, dataloader=None):
        if dataloader is None:
            dataloader = self.get_dataloader()

        data = []
        print("Start inference:")
        for idx, sample in enumerate(tqdm(dataloader)):
            input = sample[0]['input'].to(self.device)
            z = torch.randn(input.shape[0], self.config.latent_dim, input.shape[2],  input.shape[3])
            z = z.type_as(input)
            input = torch.cat([input, z], dim=1)
            yhat = self.generator(input)

            data.append(yhat.squeeze().cpu())
            if self.max_num_inference_steps is not None:
                if idx > self.max_num_inference_steps - 1:
                    break
        self.model_output = torch.cat(data)

    def get_target(self):
        dataloader = self.get_dataloader()
        target = []
        for i, batch in enumerate(tqdm(dataloader)):
            target.append(batch[0]['target'])
        target = torch.cat(target)
        return target

    def apply_inverse_transforms(self):
        for i in range(len(self.model_output)):
            data = self.model_output[i]
            data = self.transforms.inverse_normalize(data)
            data = self.transforms.inverse_log(data)
            self.model_output[i] = data
    
    def get_netcdf_result(self):
        
        time = self.cmip.sel(time=slice(self.test_start, self.test_end)).time

        if self.projection:
            time = xr.open_dataset(self.projection_path).time

        if self.max_num_inference_steps is not None:
            time = time.isel(time=slice(0, (self.max_num_inference_steps+1)*self.tst_batch_sz))

        latitude = self.cmip.latitude.values
        longitude = self.cmip.longitude.values
        np.testing.assert_array_equal(self.model_output.shape[0], len(time),
                                    'Time dimensions dont have matching shapes.') 
        np.testing.assert_array_equal(self.model_output.shape[1], len(latitude),
                                    'Latitude dimensions dont have matching shapes.') 
        np.testing.assert_array_equal(self.model_output.shape[2], len(longitude),
                                    'Longitude dimensions dont have matching shapes.') 

        gan_data= xr.DataArray(
            data=self.model_output,
            dims=["time", "latitude", "longitude"],
            coords=dict(
                time=time,
                latitude=latitude,
                longitude=longitude,
            ),
            attrs=dict(
                description="precipitation",
                units="mm/s",
            ))
        
        gan_dataset = gan_data.to_dataset(name="precipitation")
        self.gan_dataset = gan_dataset.transpose('time', 'latitude', 'longitude')

        return self.gan_dataset

