sequential:
	gcc8 -O3 -o sequential -lm main.c -march=native -mtune=native
omp:
	gcc8 -O3 -o omp -lm main_omp_optimized.c -fopenmp -march=native -mtune=native
mpi:
	mpicc -o mpi -lm main_mpi.c -fopenmp -O3 -march=native -mtune=native

mpi_optimized:
	mpicc -o mpi_optimized -lm main_mpi_optimized.c -fopenmp -O3 -march=native -mtune=native
run_sequential:
	sh -c "time ./sequential"
run_omp:
	sh -c "time ./omp 2"
	sh -c "time ./omp 4"
	sh -c "time ./omp 8"

.PHONY: sequential omp mpi mpi_optimized run_omp

run_mpi_optimized:
	sh -c "time mpirun -np 2 ./mpi_optimized 2"
	sh -c "time mpirun -np 4 ./mpi_optimized 2"
