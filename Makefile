sequential:
	gcc8 -O3 -o sequential -lm main.c -march=native -mtune=native
omp:
	gcc8 -O3 -o omp -lm main_omp_optimized.c -fopenmp -march=native -mtune=native
mpi:
	mpicc -o mpi -lm main_mpi_omp.c -fopenmp -O3 -march=native -mtune=native

.PHONY: sequential omp mpi
