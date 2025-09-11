from juliacall import Main as jl
import juliapkg
import numpy as np
juliapkg.add("ThinPlateSplines", "1d861738-f48e-4029-b1d3-81ce6bc7f5ab", url="https://github.com/mkitti/ThinPlateSplines.jl", rev="mkitti-type-stability")
juliapkg.add("ShroffCelegansModels","28a312d2-d9d3-46a7-98c1-9c09f12e8c99",path="/Users/malinmayorc/code/shroff/ShroffCelegansModels.jl", dev=True)
# juliapkg.add("ShroffCelegansModels","28a312d2-d9d3-46a7-98c1-9c09f12e8c99",url="https://github.com/JaneliaSciComp/ShroffCelegansModels.jl", dev=True)
juliapkg.resolve()

jl.seval("using ThinPlateSplines")
jl.seval("using ShroffCelegansModels")

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
julia_model = jl.ShroffCelegansModels.build_celegans_model(lattice_csv)
# print(julia_model)

point = jl.ShroffCelegansModels.Point3(3, 2, 4)
points = np.array([point])

untwisted_points = jl.ShroffCelegansModels.untwist_annotations(julia_model, jlpoints)
print(untwisted_points)