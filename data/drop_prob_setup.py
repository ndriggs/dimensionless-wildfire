from sympy.physics import units

# Setup 1: ignore all problematic variables; that is, give them units that exclude them from the nullspace of the dimensional matrix

length = units.Dimension("length")
population = units.Dimension("population")
time = units.Dimension("time")
velocity = length/time
mass = units.Dimension("mass")
energy = mass * velocity**2
unit = units.Dimension(1)
density = 1/length**2
temperature = units.Dimension("temperature")
precipitation = units.Dimension("precipitation")
water = units.Dimension("water")
impervious = units.Dimension("concrete")
fuel1 = units.Dimension("fuel1")
fuel2 = units.Dimension("fuel2")
fuel3 = units.Dimension("fuel3")

units_ = {
    'elevation': length,
    'population': population*density,
    'chili': unit,
    'pdsi': unit,
    'NDVI': unit,
    'viirs_FireMask': unit,
    'viirs_PrevFireMask': unit,
    'fuel1': fuel1,
    'fuel2': fuel2,
    'fuel3': fuel3,
    'water': water,
    'impervious': impervious,
    'erc': energy*density,  # energy release component
    'sph': unit,  # humidity
    'th': unit,  # wind direction
    'pr': precipitation, # precipitation
    'vs': velocity,  # wind speed
    'bi': unit,   # burning index
    'tmmx': temperature,
    'tmmn': temperature,
    'boltzmann': energy/temperature
}


positive=["elevation", "population", "vs", "erc"]
constants={"boltzmann": 1.380649e-23}

