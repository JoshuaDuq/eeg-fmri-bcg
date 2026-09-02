"""Study orchestration: one neutral entry point over every correction method.

The correction methods live in ``bcg_correction`` (AAS, PCA-OBS, blocked mean)
and ``bcgnet`` (the GRU). This package is the layer above them, deliberately not
named after any one of them -- running AAS through a command called ``bcgnet``
is the sort of thing that quietly tells users which method is the real one.
"""
