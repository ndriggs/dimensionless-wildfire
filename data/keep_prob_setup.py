
from sympy.physics import units

# Setup 2: include all problematic variables; that is, call them dimensionless to include them in the nullspace of the dimensional matrix

length = units.Dimension("length")
time = units.Dimension("time")
velocity = length/time
mass = units.Dimension("mass")
energy = mass * velocity**2
unit = units.Dimension(1)
density = 1/length**2
temperature = units.Dimension("temperature")

units_ = {
    'elevation': length,
    'population': unit,
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

positive = ["elevation", "population", "vs", "erc"]

constants = {"boltzmann": 1.380649e-23}
