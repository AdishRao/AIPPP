# Imports
from ferb_worker import *

nprocs = None

def ferb_init():
    global nprocs
    global my_rank

    environ['DISPLAY'] = ':0.0'

    nprocs = comm.Get_size()
    my_rank = comm.Get_rank()
    
    if(my_rank==0):
        gpu_init()
        return

    worker_main()

    exit()

def ferb_deinit():
    cmd = TERMINATE
    comm.bcast(cmd, root=0)
    return

def vec_add(A,B,n):
    
    if ((len(A)<1) or (len(B)<1) or (n < 1)):
	    return None
    
    send_prog = VEC_ADD_ID
    comm.bcast(send_prog, root=0)

    #tell every worker how much space to allocate in the recv buffer
    n_per_proc = int((n + nprocs - 1) / nprocs)
    comm.bcast(n_per_proc, root=0)

    buffered_size = int(n_per_proc * nprocs)

    # To send all the data to other nodes
    buffered_A = np.concatenate((np.copy(A),np.zeros(int(buffered_size)-abs(len(A)))))
    buffered_B = np.concatenate((np.copy(B),np.zeros(int(buffered_size)-abs(len(B)))))
    buffered_C = np.zeros(int(buffered_size))

    # To compute locally
    my_A = np.zeros(n_per_proc)
    my_B = np.zeros(n_per_proc)
    my_C = np.zeros(n_per_proc)

    # Send all the data
    comm.Scatter(buffered_A, my_A, root=0)
    comm.Scatter(buffered_B, my_B, root=0)

    # Compute the sum
    i = 0
    mod_size = n_per_proc % MAX_VEC_SIZE

    if (mod_size != 0):
        vec_add_gpu(my_A, my_B, my_C, mod_size)
        i = mod_size

    for j in range(i,n_per_proc,MAX_VEC_SIZE):
        vec_add_gpu(my_A[j:j+MAX_VEC_SIZE], my_B[j:j+MAX_VEC_SIZE], my_C[j:j+MAX_VEC_SIZE], MAX_VEC_SIZE)

    comm.Gather(my_C, buffered_C, root=0)

    comm.Barrier()

    return buffered_C

def sigmoid_activation(A,n):
    if ((len(A)<1) or (n < 1)):
        return None

    send_prog = SIGMOID_ACT_ID
    comm.bcast(send_prog, root=0)

    n_per_proc = int((n + nprocs - 1) / nprocs)
    comm.bcast(n_per_proc, root=0)

    buffered_size = int(n_per_proc * nprocs)

    buffered_A = np.concatenate((np.copy(A),np.zeros(int(buffered_size)-abs(len(A)))))
    buffered_C = np.zeros(int(buffered_size))

    # To compute locally
    my_A = np.zeros(n_per_proc)
    my_C = np.zeros(n_per_proc)

    comm.Scatter(buffered_A, my_A, root=0)

    i = 0
    mod_size = n_per_proc % MAX_VEC_SIZE

    if (mod_size != 0):
        sigmoid_gpu(my_A, my_C, mod_size)
        i = mod_size

    for j in range(i,n_per_proc,MAX_VEC_SIZE):
        sigmoid_gpu(my_A[j:j+MAX_VEC_SIZE], my_C[j:j+MAX_VEC_SIZE], MAX_VEC_SIZE)

    comm.Gather(my_C, buffered_C, root=0)

    comm.Barrier()

    return buffered_C

def tanh_activation(A, n):
    if ((len(A)<1) or (n < 1)):
        return None

    send_prog = TANH_ACT_ID
    comm.bcast(send_prog, root=0)

    n_per_proc = int((n + nprocs - 1) / nprocs)
    comm.bcast(n_per_proc, root=0)

    buffered_size = int(n_per_proc * nprocs)

    buffered_A = np.concatenate((np.copy(A),np.zeros(int(buffered_size)-abs(len(A)))))
    buffered_C = np.zeros(int(buffered_size))

    # To compute locally
    my_A = np.zeros(n_per_proc)
    my_C = np.zeros(n_per_proc)

    comm.Scatter(buffered_A, my_A, root=0)

    i = 0
    mod_size = n_per_proc % MAX_VEC_SIZE

    if (mod_size != 0):
        tanh_gpu(my_A, my_C, mod_size)
        i = mod_size

    for j in range(i,n_per_proc,MAX_VEC_SIZE):
        tanh_gpu(my_A[j:j+MAX_VEC_SIZE], my_C[j:j+MAX_VEC_SIZE], MAX_VEC_SIZE)

    comm.Gather(my_C, buffered_C, root=0)

    comm.Barrier()

    return buffered_C

def mat_mul(A, B, m, n, p):
    
    if ((len(A)<1) or (len(B)<1) or (n < 1) or (m < 1) or (p < 1)):
        return None
    send_prog = MAT_MUL_ID
    comm.bcast(send_prog, root=0)
    A_dim = m * n
    B_dim = n * p
    C_dim = m * p

    
    A_rows_per_proc = int((m + nprocs - 1) / nprocs)
    A_elems_per_proc = int(A_rows_per_proc * n)
    C_elems_per_proc = int(A_rows_per_proc * p)

    metadata = [0]*5
    metadata[0] = A_rows_per_proc
    metadata[1] = A_elems_per_proc
    metadata[2] = n
    metadata[3] = p
    metadata[4] = C_elems_per_proc
    comm.Barrier()
    comm.bcast(metadata, root=0)

    buffered_size_A = int(A_elems_per_proc * nprocs)

    buffered_A = np.zeros(buffered_size_A)
    buffered_B = np.zeros(B_dim)

    buffered_size_C = int(C_elems_per_proc * nprocs)
    buffered_C = np.zeros(buffered_size_C)

    buffered_A = np.concatenate((np.copy(A),np.zeros(int(buffered_size_A)-abs(len(A)))))
    buffered_B = np.concatenate((np.copy(B),np.zeros(int(B_dim)-abs(len(B)))))

    my_A = np.zeros(A_elems_per_proc)
    my_C = np.zeros(C_elems_per_proc)
    comm.Barrier()
    comm.Scatter(buffered_A, my_A, root=0)
    comm.Bcast(buffered_B, root=0)
    mat_mul_gpu(my_A, buffered_B, my_C, A_rows_per_proc, n, p)
    comm.Gather(my_C, buffered_C, root=0)

    buffered_A, buffered_B = None, None
    my_A, my_C = None, None
    comm.Barrier()

    return buffered_C