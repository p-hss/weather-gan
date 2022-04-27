import xarray as xr
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch

import random

from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali import pipeline_def, fn
from dask.diagnostics import ProgressBar
#import cupy as cp
import pytorch_lightning as pl
from src.data import month_from_daily_date
from torch.utils.data import DataLoader


class PyTorchDataModule(pl.LightningDataModule):
    """Main class to prepare dataloader for training. """

    def __init__(self,
                 config
                ):


        super().__init__()

        self.config = config
        self.num_workers = config.num_workers
        self.prefetch_factor = config.prefetch_factor
        self.train_batch_size = config.train_batch_size
        self.test_batch_size =  config.test_batch_size
        # order of variable lists matters!
        self.input_variables = ['precipitation', 'temperature']
        self.target_variables = ['precipitation', 'temperature']
        self.input_fname = config.input_fname
        self.target_fname = config.target_fname
        self.batch_names = ['input', 'target']
        
        self.splits = {
                "train": [str(config.train_start), str(config.train_end)],
                "valid": [str(config.valid_start), str(config.valid_end)],
                "test":  [str(config.test_start), str(config.test_end)]
        }
        

    def get_dataset(self, stage: str, fname: str, variables):
        preproc = ProcessDataset(fname,
                        variables,
                        self.splits[stage])
        
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

            dataset = DaskDataset(stage,
                                  target_dataset,
                                  input_dataset,
                                  self.target_variables,
                                  self.input_variables)

            self.train_loader = DataLoader(dataset,
                                           batch_size=self.train_batch_size,
                                           prefetch_factor=self.prefetch_factor,
                                           shuffle=True,
                                           drop_last=True,
                                           num_workers=self.num_workers)
            
            input_dataset = self.get_dataset('valid', self.input_fname, self.input_variables)
            target_dataset = self.get_dataset('valid', self.target_fname, self.target_variables)

            dataset = DaskDataset(stage,
                                  target_dataset,
                                  input_dataset,
                                  self.target_variables,
                                  self.input_variables)

            self.valid_loader =  DataLoader(dataset,
                                           batch_size=self.test_batch_size,
                                           prefetch_factor=self.prefetch_factor,
                                           shuffle=False,
                                           drop_last=True,
                                           num_workers=self.num_workers)

        if stage == 'test':
            input_dataset = self.get_dataset('test', self.input_fname, self.input_variables)
            target_dataset = self.get_dataset('test', self.target_fname, self.target_variables)

            dataset = DaskDataset(stage,
                                  target_dataset,
                                  input_dataset,
                                  self.target_variables,
                                  self.input_variables)

            self.test_loader = DataLoader(dataset,
                                           batch_size=self.test_batch_size,
                                           prefetch_factor=self.prefetch_factor,
                                           shuffle=False,
                                           drop_last=False,
                                           num_workers=self.num_workers)


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
                 variables,
                 time_slice,
                 chunk_size=10):
    
        #self.ds = xr.open_dataset(fname, chunks={"time": chunk_size})
        #self.ds = xr.open_dataset(fname).load()
        print("loading datasets..")
        self.ds = xr.open_dataset(fname).load()
        print("finished")
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


class DaskDataset(torch.utils.data.Dataset):
    
    def __init__(self,
                 stage,
                 target_dataset,
                 input_dataset,
                 target_variables,
                 input_variables,
                 ):

        self.target_dataset = target_dataset
        self.target_variables = target_variables
        self.input_dataset = input_dataset
        self.input_variables = input_variables
        self.stage = stage
        self.time_axis = target_dataset.time


    def pack_input_batch(self, index):

        channel = []
        for var in self.input_variables:

            date = month_from_daily_date(self.time_axis, index)
            image = self.input_dataset[var].sel(time=date)
            image_tensor = image.values.squeeze()
            channel.append(image_tensor)
            
        channel_stacked = np.stack(channel, axis=0)
        #channel_stacked = np.expand_dims(channel_stacked, axis=0)
        return channel_stacked


    def pack_target_batch(self, index):

        channel = []
        for var in self.target_variables:
            image = self.target_dataset[var].isel(time=index)
            image_tensor = image.values.squeeze()
            channel.append(image_tensor)
            
        channel_stacked = np.stack(channel, axis=0)
        #channel_stacked = np.expand_dims(channel_stacked, axis=0)
        return channel_stacked
                
        
    def __getitem__(self, index):
        
        inputs = self.pack_input_batch(index)

        # add transforms below:
        inputs = Transforms().log(inputs)
        inputs = Transforms().normalize(inputs)
        inputs = Transforms().crop(inputs)
    
        targets = self.pack_target_batch(index)
        # add transforms below:
        if self.stage != 'test':
            targets = Transforms().log(targets)
            targets = Transforms().normalize(targets)
        targets = Transforms().crop(targets)
        
        return torch.from_numpy(inputs), torch.from_numpy(targets)

    def __len__(self):
        return len(self.target_dataset.time) 


class Transforms():
    
    def __init__(self, min_ref=0, max_ref=4):
        self.epsilon = 0.0001
        self.temperature_min_ref = 190
        self.temperature_max_ref = 320

        self.log_precipitation_min_ref = 0 
        self.log_precipitation_max_ref = 4

        self.log_function = np.log
        self.exp_function = np.exp
        self.scale_factor = 100
        
    def crop(self, x):
        return x[:,:-1,:-72]

    def abs(self, x):
        return np.abs(x)
    
    def log(self, x):
        """applies log transform to one variable only """
        x[0,:,:] = self.log_function(x[0,:,:] + self.epsilon) - self.log_function(self.epsilon)
        return  x    

    def inverse_log(self, x):
        x[0,:,:] = self.exp_function(x[0,:,:] + self.log_function(self.epsilon)) - self.epsilon
        return x

    def normalize(self, x):
        """ normalize to [-1, 1] """
        x[0,:,:] = (x[0,:,:] - self.log_precipitation_min_ref)/(self.log_precipitation_max_ref - self.log_precipitation_min_ref)
        x[1,:,:] = (x[1,:,:] - self.temperature_min_ref)/(self.temperature_max_ref - self.temperature_min_ref)
        x = x*2 - 1
        return x
    
    def inverse_normalize(self, x):
        x = (x + 1)/2
        x[0,:,:] = x[0,:,:]*(self.log_precipitation_max_ref - self.log_precipitation_min_ref) + self.log_precipitation_min_ref
        x[1,:,:] = x[1,:,:]*(self.temperature_max_ref - self.temperature_min_ref) + self.temperature_min_ref
        return x 
    
    def test(self):
        x = np.ones((10,10))
        x_ref = x
        x = self.log(x)
        x = self.normalize(x)
        x = self.inverse_normalize(x)
        x = self.inverse_log(x)
        np.testing.assert_array_equal(x, x_ref)



