import juliapkg
juliapkg.add("ThinPlateSplines", "1d861738-f48e-4029-b1d3-81ce6bc7f5ab", url="https://github.com/mkitti/ThinPlateSplines.jl", rev="mkitti-type-stability")
juliapkg.add("ShroffCelegansModels","28a312d2-d9d3-46a7-98c1-9c09f12e8c99",url="https://github.com/JaneliaSciComp/ShroffCelegansModels.jl", dev=True)
juliapkg.resolve()
print("Julia environment configured")
