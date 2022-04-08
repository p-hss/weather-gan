import xarray as xr
import torch
import numpy as nmp

from src.model import WeatherGenerator

class Inference():

    """ Execute model on test data and return output as NetCDF. """
    
    def __init__(self,
                 config,
                 constrain=False,
                 projection=False,
                 projection_path=None,
                 max_num_inference_steps=None):
        

        self.config = config
        scratch_path: str = '/p/tmp/hess/scratch/cmip-gan'
        self.results_path = config.results_path

        self.train_start = str(config.train_start)
        self.train_end = str(config.train_end)
        self.test_start = str(config.test_start)
        self.test_end = str(config.test_end)

        self.model = None
        self.model_output = None
        self.dataset = None

        self.transforms = config.transforms
        self.max_num_inference_steps = max_num_inference_steps
        self.tst_batch_sz = 64
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        
    def get_config(self, config_path):
        
    def load_model(self, checkpoint_path):
    
        model = WeatherGenerator(self.config).load_from_checkpoint(checkpoint_path=checkpoint_path)
        model.freeze()
        self.model = model.to(self.device)
        self.model = ConstrainedGenerator(self.model.g_B2A, constrain=self.constrain)
            
    def compute(self, dataloader=None):
        """ Use B (ESM) -> A (ERA5) generator for inference """
        if dataloader is None:
            test_data = self.get_dataloader()
        else:
            test_data = dataloader

        data = []

        print("Start inference:")
        for idx, sample in enumerate(tqdm(test_data)):
            sample = sample['B'].to(self.device)
            yhat = self.model(sample)

            data.append(yhat.squeeze().cpu())
            if self.max_num_inference_steps is not None:
                if idx > self.max_num_inference_steps - 1:
                    break
            
        self.model_output = torch.cat(data)


    def test(self):
        dataset = CycleDataset('test', self.config)
        test_data = dataset[0]
        sample = test_data['A'][0]
        data = self.inv_transform(sample)
        print(data.min(), data.max())

    
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
