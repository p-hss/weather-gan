import xarray as xr
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch

from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali import pipeline_def, fn
from dask.diagnostics import ProgressBar
import cupy as cp
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    """Main class to prepare dataloader for training. """

    def __init__(self,
                 #config,
                 num_workers=0,
                 train_batch_size: int = 4,
                 test_batch_size: int = 64):

        """
        Parameters:
            train_batch_size: Training Batch Size
            test_batch_size: Test Batch Size
        """

        super().__init__()

        #self.config = config
        #self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.input_variables = ['precipitation', 'precipitation_2']
        self.target_variables = ['precipitation', 'precipitation_2']
        self.input_fname = '/home/ftei-dsw/data/dataloader/cmip_2.nc'
        self.target_fname = '/home/ftei-dsw/data/dataloader/cmip_2.nc'
        self.batch_names = ['input', 'target']
        
        self.splits = {
        
                'train': ['2000', '2001'],
                'valid': ['2002', '2003'],
                'test': ['2004', '2005']
        }
        

    def get_dataset(self, stage: str, fname: str, variables):
        preproc = ProcessDataset(fname,
                        variables=variables,
                        time_slice=self.splits[stage])
        
        preproc.run()
        dataset = preproc.get()
        return dataset
    

    def setup(self, stage: str = None):

        """
        stage: fit / test
        """

        if stage == 'fit' or stage is None:
            input_dataset = self.get_dataset('train', self.input_fname, self.input_variables)
            target_dataset = self.get_dataset('train', self.target_fname, self.target_variables)
            self.train_loader = DaliLoader(input_dataset,
                                           target_dataset,
                                           self.train_batch_size,
                                           self.batch_names,
                                           shuffle=True)
            
            input_dataset = self.get_dataset('valid', self.input_fname, self.input_variables)
            target_dataset = self.get_dataset('valid', self.target_fname, self.target_variables)
            self.valid_loader = DaliLoader(input_dataset,
                                           target_dataset,
                                           self.test_batch_size,
                                           self.batch_names,
                                           shuffle=True)

        if stage == 'test':
            input_dataset = self.get_dataset('test', self.input_fname, self.input_variables)
            target_dataset = self.get_dataset('test', self.target_fname, self.target_variables)
            self.test_loader = DaliLoader(input_dataset,
                                           target_dataset,
                                           self.test_batch_size,
                                           self.batch_names,
                                           shuffle=True)


    def train_dataloader(self):
        return self.train_loader


    def valid_dataloader(self):
        return self.valid_loader

    
    def test_dataloader(self):
        return self.test_loader


class ProcessDataset():
    """ Preprocess the NetCDF dataset:
            - select variables
            - select time interval
    """
    def __init__(self,
                 fname: str,
                 variables=['precipitation', 'precipitation_2'],
                 time_slice=['2000', '2001'],
                 chunk_size=10):
    
        
        self.ds = xr.open_dataset(fname, chunks={"time": chunk_size})
        self.variables = variables
        self.time_slice = time_slice
        self.data = None
        
        
    def run(self):
        
        data = []
        for var in self.variables:
            data.append(self.ds[var])
        self.data = xr.merge(data)
        self.data = self.data.sel(time=slice(self.time_slice[0], self.time_slice[1]))
        
        
    def get(self):
        return self.data
    
    def get_length(self) -> int:
        return len(self.data.time)



class DaliLoader():
    """ Dataloader used in the PyTorch training loop """

    def __init__(self, input_dataset, target_dataset, batch_size, output_map, shuffle=True):
        
        
        self.length = len(input_dataset.time)
        
        input_source = ExternalInputIterator(input_dataset, batch_size, shuffle=shuffle)
        target_source = ExternalInputIterator(target_dataset, batch_size, shuffle=shuffle)

        pipe = pipeline(input_source,
                        target_source,
                        batch_size=batch_size,
                        num_threads=12,
                        device_id=0,
                        exec_async=False,
                        exec_pipelined=False)
        
        self.dali_iterator = DALIGenericIterator(pipe, output_map, auto_reset=True) 
        
    def __len__(self):
        return int(self.length)
    
    def __iter__(self):
        return self.dali_iterator.__iter__()



class ExternalInputIterator(object):
    """ 
        Samples batches from Xarray dataset.
        Used in the Pipeline definition to load the data.
        
    """
    
    def __init__(self,
                 dataset: xr.Dataset,
                 batch_size: int,
                 shuffle=True
                ):
        
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.variables = ['precipitation', 'precipitation_2']
        self.length = len(dataset.time)
        self.dataset = dataset
        self.shuffle = shuffle

    def __iter__(self):
        self.i = 0
        self.n = self.length
        return self

    def __next__(self):
        batch = []
        for _ in range(self.batch_size):
            channel = []
            
            for var in self.variables:
                image = self.dataset[var].isel(time=self.i)
                #image = torch.from_numpy(image.values).to(self.device)
                image_tensor = torch.from_numpy(image.values)
                channel.append(image_tensor)
                
            channel_stacked = torch.stack(channel, dim=-1)
            channel_stacked = channel_stacked.unsqueeze(0)
            batch.append(channel_stacked)
            
            if self.shuffle is True:
                self.i = np.random.randint(0,self.n,size=1)[0]
            else:
                self.i = (self.i + 1) % self.n
            
        sample = torch.concat(batch)
        #sample = sample.unsqueeze(-1) # create a channel
        sample = sample.to(self.device)
        
        return sample

def random_date(monthly_times: xr.DataArray, index) -> str:
    month = str(monthly_times.isel(time=index)['time.month'].values) 
    year = str(monthly_times.isel(time=index)['time.year'].values)
    random_day = str(np.random.randint(1,30,size=1)[0])
    date_str = f'{year}-{month.zfill(2)}-{random_day.zfill(2)}'
    return date_str

@pipeline_def
def pipeline(input_source, target_source):
    """ Pipelines for input and target """
     
    inputs = fn.external_source(source=input_source, layout="CHW", device="gpu")
    # add transforms below:
    inputs = fn.python_function(inputs, function=Transforms().crop)
    inputs = fn.python_function(inputs, function=Transforms().scale)
    
    targets = fn.external_source(source=target_source, layout="CHW", device="gpu")
    # add transforms below:
    targets = fn.python_function(targets, function=Transforms().crop)
    return inputs, targets


class Transforms():
    
    def __init__(self, min_ref=0, max_ref=4):
        self.epsilon = 0.0001
        self.min_ref = min_ref 
        self.max_ref = max_ref
        self.log = np.log
        self.exp = np.exp
        self.scale_factor = 100
        
    def crop(self, x):
        return x[:-5,:-50]
    
    def scale(self, x):
        return x*self.scale_factor
    
    def log(self, x):
        return self.log(x + self.epsilon) - self.log(self.epsilon)
    
    def inverse_log(self, x):
        return self.exp(x + self.log(self.epsilon)) - self.epsilon

    def normalize(self, x):
        """ normalize to [-1, 1] """
        results = (x - self.min_ref)/(self.max_ref - self.min_ref)
        results = results*2 - 1
        return results
    
    def inverse_normalize(self, x):
        x = (x + 1)/2
        results = x * (self.max_ref - self.min_ref) + self.min_ref
        return results
    
    def test(self):
        x = np.ones((10,10))
        x_ref = x
        x = self.log(x)
        x = self.normalize(x)
        x = self.inverse_normalize(x)
        x = self.inverse_log(x)
        np.testing.assert_array_equal(x, x_ref)



