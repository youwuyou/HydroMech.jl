using GeoParams

# define the linear creep law
x1 = LinearViscous(; η=1e18Pa * s)

# define reference GeoUnits
CharUnits_GEO = GEO_units(; viscosity=1e19, length=1000km)

# nondimensionalize
x1_ND = nondimensionalize(x1, CharUnits_GEO)


# dimensionalized output
args = (;)
compute_εII(x1, 1e6Pa, args)
compute_τII(x1, 1e-13 / s, args) 


# non-dimensionalized output
compute_εII(x1_ND, 1e6, args)
compute_τII(x1_ND, 1e0, args)