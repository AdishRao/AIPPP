# Imports
from constants import *

def worker_main():
    gpu_init()
    program_id = 0
    progs = {VEC_ADD_ID:vec_add_worker, TERMINATE: exit, SIGMOID_ACT_ID:sigmoid_activation_worker, TANH_ACT_ID:tanh_activation_worker, MAT_MUL_ID:mat_mul_worker}
    while(True):
        program_id = comm.bcast(program_id)
        try:
            progs[program_id]()
        except: 
            exit()
        comm.Barrier()


def vec_add_worker():
    my_data_sz = 0
    my_data_sz = comm.bcast(my_data_sz)
    my_A = np.zeros(my_data_sz)
    my_B = np.zeros(my_data_sz)
    my_C = np.zeros(my_data_sz)

    comm.Scatter(None, my_A)
    comm.Scatter(None, my_B)

    i = 0
    mod_size = my_data_sz % MAX_VEC_SIZE
    if (mod_size != 0):
        vec_add_gpu(my_A, my_B, my_C, mod_size)
        i = mod_size

    for j in range(i,my_data_sz,MAX_VEC_SIZE):
        vec_add_gpu(my_A[j:j+MAX_VEC_SIZE], my_B[j:j+MAX_VEC_SIZE], my_C[j:j+MAX_VEC_SIZE], MAX_VEC_SIZE)

    comm.Gather(my_C, None)
    return

def sigmoid_activation_worker():
    my_data_sz = 0
    my_data_sz = comm.bcast(my_data_sz)

    my_A = np.zeros(my_data_sz)
    my_C = np.zeros(my_data_sz)

    comm.Scatter(None, my_A)

    i = 0
    mod_size = my_data_sz % MAX_VEC_SIZE

    if (mod_size != 0):
        sigmoid_gpu(my_A, my_C, mod_size)
        i = mod_size

    for j in range(i,my_data_sz,MAX_VEC_SIZE):
        sigmoid_gpu(my_A[j:j+MAX_VEC_SIZE], my_C[j:j+MAX_VEC_SIZE], MAX_VEC_SIZE)

    comm.Gather(my_C, None)
    return

def tanh_activation_worker():
    my_data_sz = 0
    my_data_sz = comm.bcast(my_data_sz)

    my_A = np.zeros(my_data_sz)
    my_C = np.zeros(my_data_sz)

    comm.Scatter(None, my_A)

    i = 0
    mod_size = my_data_sz % MAX_VEC_SIZE

    if (mod_size != 0):
        tanh_gpu(my_A, my_C, mod_size)
        i = mod_size

    for j in range(i,my_data_sz,MAX_VEC_SIZE):
        tanh_gpu(my_A[j:j+MAX_VEC_SIZE], my_C[j:j+MAX_VEC_SIZE], MAX_VEC_SIZE)

    comm.Gather(my_C, None)
    return

def mat_mul_worker():
    metadata = []
    comm.Barrier()
    metadata = comm.bcast(metadata)

    A_rows_per_proc = metadata[0]
    A_elems_per_proc = metadata[1]
    B_row = metadata[2]
    B_col = metadata[3]
    C_elems_per_proc = metadata[4]

    B_dims = B_row * B_col

    my_A = np.zeros(A_elems_per_proc)
    my_B = np.zeros(B_dims)
    my_C = np.zeros(C_elems_per_proc)
    comm.Barrier()
    comm.Scatter(None, my_A)
    comm.Bcast(my_B)
    mat_mul_gpu(my_A, my_B, my_C, A_rows_per_proc, B_row, B_col)
    comm.Gather(my_C, None)

    my_A, my_B, my_C = None, None, None
    return