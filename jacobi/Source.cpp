#include <math.h>
#include <iostream>
#include <fstream>
#include "mpi.h"
#include <Windows.h>
#include <sstream>
#include <ctime>

// #define DEBUG

// Doesn't work when size > 2*m
// TODO: Handle matrix restruct (ask Max)

using namespace std;

void generateMatrixFile(int size) {
	srand(unsigned(time(0)));
	double** matrix = new double* [size];
	for (int i = 0; i < size; i++) {
		matrix[i] = new double[size + 1];
		matrix[i][i] = 0;
		for (int j = 0; j < size + 1; j++) {
			if (i != j) {
				matrix[i][j] = rand() / 100;
				matrix[i][i] += matrix[i][j];
			}
		}
	}

	std::ofstream matrixOutput;
	std::ofstream approxOutput;
	matrixOutput.open("C:\\university\\multithreading\\visual_studio\\jacobi\\jacobi\\matrix_gen.txt");
	approxOutput.open("C:\\university\\multithreading\\visual_studio\\jacobi\\jacobi\\approx_gen.txt");

	matrixOutput << size << " " << size + 1 << endl;
	approxOutput << size << endl;
	for (int i = 0; i < size; i++) {
		matrixOutput << matrix[i][0];
		approxOutput << rand() / 100 << endl;
		for (int j = 1; j < size + 1; j++) {
			matrixOutput << " " << matrix[i][j];
		}
		matrixOutput << endl;
	}
}

int main(int argc, char** argv) {
	
	if (argc == 2) {
		char* input = argv[1];
		int fill = atoi(input);
		generateMatrixFile(fill);
		return 0;
	}

#ifdef FILL
	generateMatrixFile(FILL);
	return 0;
#endif


	// MPI Initialization
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	double* data;
	double** matrix;
	double** scatteredMatrix;
	double* free;
	double* scatteredFree;
	double* approx;
	double* tempApprox;
	double* norms;
	int* lengths;
	int length;
	int m, n; 
	double eps;

	std::ifstream matrixInput;
	std::ifstream approxInput;
	std::ifstream epsInput;
	std::ofstream solutionOutput;

	if (rank == 0) {
		matrixInput.open("C:\\university\\multithreading\\visual_studio\\jacobi\\jacobi\\matrix_gen.txt");
		approxInput.open("C:\\university\\multithreading\\visual_studio\\jacobi\\jacobi\\approx_gen.txt");
		epsInput.open("C:\\university\\multithreading\\visual_studio\\jacobi\\jacobi\\eps.txt");

		epsInput >> eps;
		epsInput.close();

		matrixInput >> m >> n;
		approxInput >> m;

		if (m % size == 0) n = m;
		else if (m > size) n = m + size - (m % size);
		else               n = size;
	}

	MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// matrix, free and approx init
	data = (double*)malloc(n * m * sizeof(double));
	matrix = (double**)malloc(n * sizeof(double*));
	for (int i = 0; i < n; i++) matrix[i] = &(data[m * i]);
	free = (double*)malloc(n * sizeof(double));
	approx = (double*)malloc(n * sizeof(double));

	if (rank == 0) {
		// read matrix, free and approx from files
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < m; j++) {
				matrixInput >> matrix[i][j];
			}
			matrixInput >> free[i];
		}

		for (int i = 0; i < m; i++) {
			approxInput >> approx[i];
		}

		matrixInput.close();
		approxInput.close();
	}

	// temp approx init
	tempApprox = (double*)malloc((n / size) * sizeof(double));
	// scattered matrix, free init
	data = (double*)malloc(m * (n / size) * sizeof(double));
	scatteredMatrix = (double**)malloc((n / size) * sizeof(double*));
	for (int i = 0; i < n / size; i++) scatteredMatrix[i] = &(data[m * i]);
	scatteredFree = (double*)malloc(n * sizeof(double));
	// norms init
	norms = (double*)malloc(size * sizeof(double));
	// sizes init
	lengths = (int*)malloc(size * sizeof(int));
	for (int i = 0; i < size; i++) {
		lengths[i] = n / size;
	}
	// last is n/size - (n-m)
	lengths[size - 1] += m - n;

	// start itme
	double startwtime = MPI_Wtime();

	// initial scatter and broadcast
	MPI_Scatter(lengths, 1, MPI_INT, &length, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatter(*matrix, (n / size) * m, MPI_DOUBLE, *scatteredMatrix, (n / size) * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(free, n / size, MPI_DOUBLE, scatteredFree, n / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(approx, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(norms, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// local norm
	double norm;

	// jacobi run
	do {
		for (int i = 0; i < length; i++) {
			tempApprox[i] = scatteredFree[i];
			for (int j = 0; j < m; j++) {
				if (i + (n / size) * rank != j) {
					tempApprox[i] -= scatteredMatrix[i][j] * approx[j];
				}
			}
			tempApprox[i] /= scatteredMatrix[i][i + (n / size) * rank];
		}
		norm = fabs(approx[rank * (n / size)] - tempApprox[0]);
		for (int i = 1; i < length; i++) {
			if (fabs(approx[rank * (n / size) + i] - tempApprox[i]) > norm)
				norm = fabs(approx[rank * (n / size) + i] - tempApprox[i]);
			approx[rank * (n / size) + i] = tempApprox[i];
		}

		// midterm gather and broadcast
		MPI_Gather(tempApprox, n / size, MPI_DOUBLE, approx, n / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Gather(&norm, 1, MPI_DOUBLE, norms, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(approx, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(norms, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// getting max norm
		norm = norms[0];
		for (int i = 0; i < size; i++) {
			if (norms[i] > norm) {
				norm = norms[i];
			}
		}
	} while (norm > eps);
	
	delete[] tempApprox;
	
	// end time
	double endwtime = MPI_Wtime();

	MPI_Finalize();

	if (rank == 0) {
		cout << endwtime - startwtime << endl;
	}
	
	// print solution
	if (rank == 0) {
		solutionOutput.open("C:\\university\\multithreading\\visual_studio\\jacobi\\jacobi\\solution.txt");
		for (int i = 0; i < m; i++) {
			solutionOutput << approx[i] << endl;
		}
	}

	delete[] data;
	delete[] matrix;
	delete[] scatteredMatrix;
	delete[] free;
	delete[] scatteredFree;
	delete[] approx;
	delete[] tempApprox;
	delete[] norms;
	delete[] lengths;

	return 0;
}