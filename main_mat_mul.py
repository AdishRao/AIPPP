# Import Statements
import time
from ferb_master import *

def mat_mul_cpu(A, B, m, n, p):
    C = np.zeros(m*p)

    for a_row in range(m):
        for b_col in range(p):
            for a_col in range(n):
                C[(a_row*p)+b_col] += A[(a_row*n)+a_col] * B[(a_col*p)+b_col]
    return C

ferb_init()

m = int(argv[1])
n = int(argv[2])
p = int(argv[3])

A = np.random.uniform(-5.0,5.0,[m*n])
B = np.random.uniform(-5.0,5.0,[n*p])

start_time = time.time()

C = mat_mul(A, B, m, n, p)

end_time = time.time()

ferb_deinit()

C = C[:len(A)]

res = mat_mul_cpu(A, B, m, n, p)

error = (sum(pow((C - res),2))/(m * p))**(0.5)

print(m, n, p, argv[4], argv[5], error, (end_time-start_time), sep=',')

