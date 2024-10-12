
from torch.utils.data import DataLoader
import torch
from notebooks import dataloader_test as dlt
from sympy.physics import units


length = units.Dimension("length")
population = units.Dimension("population")
time = units.Dimension("time")
velocity = length/time
mass = units.Dimension("mass")
energy = mass * velocity**2
unit = units.Dimension(1)
density = 1/length**2
temperature = units.Dimension("temperature")

units_ = {
    'elevation': length,
    'population': population*density,
    'chili': unit,
    'pdsi': unit,
    'NDVI': unit,
    'viirs_FireMask': unit,
    'viirs_PrevFireMask': unit,
    'fuel1': unit,
    'fuel2': unit,
    'fuel3': unit,
    'water': unit,
    'impervious': unit,
    'erc': energy*density,  # energy release component
    'sph': unit,  # humidity
    'th': unit,  # wind direction
    'pr': unit, # precipitation
    'vs': velocity,  # wind speed
    'bi': unit,   # burning index
    'tmmx': temperature,
    'tmmn': temperature,
    'boltzmann': energy/temperature
}





train_files = [f'../data/modified_ndws/train_conus_west_ndws_0{i:02}.tfrecord' for i in range(39)]
test_files = [f'../data/modified_ndws/test_conus_west_ndws_0{i:02}.tfrecord' for i in range(39)]
val_files = [f'../data/modified_ndws/eval_conus_west_ndws_0{i:02}.tfrecord' for i in range(39)]



dataset = dlt.NondimFireDataset(train_files, units_, positive=["elevation", "population", "vs"], constants={"boltzmann": 1})
loader = dlt.DataLoader(dataset, batch_size=32)



