from juliacall import Main as jl
import numpy as np
jl.seval("using ShroffCelegansModelsCore")

# x1 = np.array([[0.0, 1.0,],[1.0, 0.0,],[1.0, 1.0,]])
# x2 = np.array([[0.0, 1.0,],[1.0, 0.0,],[1.2, 1.5,]])

# tps = jl.tps_solve(x1,x2,1.0)
# x3 = np.array([[0.0, 1.0,],[2.0, 2.0,]])
# deformed = jl.tps_deform(x2,tps)
# jl.println(deformed)
# print(deformed, type(deformed))
# print(np.array(deformed))

# model = jl.ShroffCelegansModel()

lattice_csv = "lattice.csv"
julia_model = jl.ShroffCelegansModelsCore.build_celegans_model(lattice_csv)
# print(julia_model)

point = jl.ShroffCelegansModelsCore.Point3(3, 2, 4)

# points = np.array([[3,2,4]])
# point = np.array((3,2,4))

untwisted_points = jl.ShroffCelegansModelsCore.untwist_annotation(julia_model, point)
print(untwisted_points)