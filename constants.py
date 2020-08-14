from mpi4py import MPI
import numpy as np
from sys import argv
from os import environ
from ferb_gpu import *

comm = MPI.COMM_WORLD
MAX_VEC_SIZE = 1000
my_rank = None