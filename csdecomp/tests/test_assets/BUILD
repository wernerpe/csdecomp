filegroup(
    name = "urdf_files",
    srcs = glob(["**/*.urdf"]) + 
           glob(["**/*.obj"]) + 
           glob(["**/*.STL"]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "directives",
    srcs = glob(["directives/*.yml"]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "voxel_grids",
    srcs = ["random_grid.voxels.txt"],
    visibility = ["//visibility:public"],
)
