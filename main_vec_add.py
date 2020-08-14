# Import Statements
import time
from ferb_master import *

# Initalize the cluster
ferb_init()

# Get number of vectors
n_vec = int(argv[1])

# Create both Arrays
A = np.random.uniform(-256,256,[n_vec])
B = np.random.uniform(-256,256,[n_vec])

start_time = time.time()

# Calculation
C = vec_add(A, B, n_vec)

end_time = time.time()

# Deinit the cluster
ferb_deinit()

C = C[:len(A)]

# Calculate the error
error = (sum(pow((C - (A+B)),2))/n_vec)**(0.5)

# Print and Save results
print(n_vec,argv[2],argv[3],error,(end_time-start_time), sep=',')
