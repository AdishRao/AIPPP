# Import Statements
import time
from ferb_master import *

ferb_init()

n_vec = int(argv[1])

A = np.random.uniform(-5.0,5.0,[n_vec])

start_time = time.time()

C = sigmoid_activation(A, n_vec)

end_time = time.time()

ferb_deinit()

C = C[:len(A)]

error = (sum(pow((C - (1/(1+np.exp(-A)))),2))/n_vec)**(0.5)

print(n_vec,argv[2],argv[3],error,(end_time-start_time), sep=',')