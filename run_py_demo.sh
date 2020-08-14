#!/bin/bash

ssh fa@192.168.1.52 'mkdir -p ~/FYP_Code/'
ssh fa@192.168.1.53 'mkdir -p ~/FYP_Code/'
ssh fa@192.168.1.54 'mkdir -p ~/FYP_Code/'
ssh fa@192.168.1.55 'mkdir -p ~/FYP_Code/'
ssh fa@192.168.1.56 'mkdir -p ~/FYP_Code/'
ssh fa@192.168.1.57 'mkdir -p ~/FYP_Code/'
ssh fa@192.168.1.58 'mkdir -p ~/FYP_Code/'

scp ~/FYP_Code/* fa@192.168.1.52:~/FYP_Code/
scp ~/FYP_Code/* fa@192.168.1.53:~/FYP_Code/
scp ~/FYP_Code/* fa@192.168.1.54:~/FYP_Code/
scp ~/FYP_Code/* fa@192.168.1.55:~/FYP_Code/
scp ~/FYP_Code/* fa@192.168.1.56:~/FYP_Code/
scp ~/FYP_Code/* fa@192.168.1.57:~/FYP_Code/
scp ~/FYP_Code/* fa@192.168.1.58:~/FYP_Code/

echo "--------------------"
echo "Binaries distributed"
echo "--------------------"
echo ""

echo "-----------------------------"
echo "Processing vec_add"
echo "-----------------------------"

echo "no of vectors,no of procs,itter,error,time" >> results_add.csv
echo "no of vectors,no of procs,itter,error,time" >> results_sigmoid.csv
echo "no of vectors,no of procs,itter,error,time" >> results_tanh.csv
echo "m,n,p,no of procs,itter,error,time" >> results_mul.csv

for n_vec in {100,1000}
do
	echo "--------------------"
	echo "vector size: $n_vec"
	echo "--------------------"
	
	for procs in {1..8}
	do
		echo "$procs processes"
		for i in {1..5}
		do
			echo -n "-"
			mpiexec.hydra -n $procs -hosts 192.168.1.51,192.168.1.52,192.168.1.53,192.168.1.54,192.168.1.55,192.168.1.56,192.168.1.57,192.168.1.58 python ~/FYP_Code/main_vec_add.py $n_vec $procs $i >> ~/FYP_Code/results_add.csv
		done
		echo
	done
done

echo "-----------------------------"
echo "Processing Sigmoid"
echo "-----------------------------"

for n_vec in {100,1000}
do
	echo "--------------------"
	echo "vector size: $n_vec"
	echo "--------------------"
	
	for procs in {1..8}
	do
		echo "$procs processes"
		for i in {1..5}
		do
			echo -n "-"
			mpiexec.hydra -n $procs -hosts 192.168.1.51,192.168.1.52,192.168.1.53,192.168.1.54,192.168.1.55,192.168.1.56,192.168.1.57,192.168.1.58 python ~/FYP_Code/main_sigmoid.py $n_vec $procs $i >> ~/FYP_Code/results_sigmoid.csv
		done
		echo
	done
done

echo "-----------------------------"
echo "Processing tanh"
echo "-----------------------------"

for n_vec in {100,1000}
do
	echo "--------------------"
	echo "vector size: $n_vec"
	echo "--------------------"
	
	for procs in {1..8}
	do
		echo "$procs processes"
		for i in {1..5}
		do
			echo -n "-"
			mpiexec.hydra -n $procs -hosts 192.168.1.51,192.168.1.52,192.168.1.53,192.168.1.54,192.168.1.55,192.168.1.56,192.168.1.57,192.168.1.58 python ~/FYP_Code/main_tanh.py $n_vec $procs $i >> ~/FYP_Code/results_tanh.csv
		done
		echo
	done
done

echo "-----------------------------"
echo "Processing Mult"
echo "-----------------------------"

for n_vec in {5,10}
do
	echo "--------------------"
	echo "vector size: $n_vec"
	echo "--------------------"
	
	for procs in {1..8}
	do
		echo "$procs processes"
		for i in {1..5}
		do
			echo -n "-"
			mpiexec.hydra -n $procs -hosts 192.168.1.51,192.168.1.52,192.168.1.53,192.168.1.54,192.168.1.55,192.168.1.56,192.168.1.57,192.168.1.58 python ~/FYP_Code/main_mat_mul.py $n_vec $n_vec $n_vec $procs $i >> ~/FYP_Code/results_mul.csv
		done
		echo
	done
done

echo "That's all folks"