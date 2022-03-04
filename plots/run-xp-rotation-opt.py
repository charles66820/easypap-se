#!/usr/bin/env python3
from graphTools import *
from expTools import *
import os

easyspap_options = {}
easyspap_options["--kernel "] = ["rotation90"]
easyspap_options["--iterations "] = [100]
easyspap_options["--variant "] = ["omp_tiled",
                                  "omp_tiled_opt"]
easyspap_options["--tile-size "] = [16, 32]
easyspap_options["--size "] = [512, 2048, 4096]
easyspap_options["-of "] = ["rotation_opt.csv"]

omp_icv = {}  # OpenMP Internal Control Variables
omp_icv["OMP_NUM_THREADS="] = [1] + list(range(2, 13, 2))
omp_icv["OMP_SCHEDULE="] = ["static"]

execute('./run', omp_icv, easyspap_options, nbrun=1)
