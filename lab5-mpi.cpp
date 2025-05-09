//#include <mpi.h>
//#include <stdio.h>
//
//#define n 4
//
//int main(int argc, char* argv[])
//{
//	int myrank, nprocs, i, j, k, map[n];
//	double a[n][n];
//	FILE* f;
//
//	MPI_Init(&argc, &argv);
//	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
//	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
//
//	if (myrank == 0)
//	{
//		f = fopen("1u_4.txt", "r");
//
//		if (f == NULL)
//		{
//			printf("Error: Unable to open file '1u_4.txt'\n");
//			MPI_Abort(MPI_COMM_WORLD, 1);
//		}
//
//		for (i = 0; i < n; i++)
//			for (j = 0; j < n; j++)	
//				fscanf(f, "%lf", &a[i][j]);
//		fclose(f);
//
//		//printf("Matrix read by process 0:\n");
//		//for (int i = 0; i < n; i++)
//		//{
//		//	for (int j = 0; j < n; j++)
//		//		printf("%lg ", a[i][j]);
//		//	printf("\n");
//		//}
//	}
//	MPI_Bcast(a, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//	for (i = 0; i < n; i++)
//		map[i] = i % nprocs;
//
//	for (k = 0; k < n - 1; k++)
//	{
//		if (map[k] == myrank)
//			for (i = k + 1; i < n; i++)
//				a[k][i] /= a[k][k];
//		MPI_Bcast(&a[k][k + 1], n - k - 1, MPI_DOUBLE, map[k], MPI_COMM_WORLD);
//		for (i = k + 1; i < n; i++)
//			if (map[i] == myrank)
//				for (j = k + 1; j < n; j++)
//					a[i][j] -= a[i][k] * a[k][j];
//	}
//
//	for (i = 0; i < n; i++)
//		if (map[i] == myrank)
//		{
//			for (j = 0; j < n; j++)
//				printf("a[%d][%d] = %lg, ", i, j, a[i][j]);
//			printf("\n");
//		}
//	printf("\n");
//	MPI_Finalize();
//	return 0;
//}



#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double hilbert_value(int i, int j) {
    return 1.0 / (i + j + 1);
}

void multiply_lu(double* l, double* u, double* result, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            result[i * n + j] = 0.0;
            for (int k = 0; k <= (i < j ? i : j); k++)
                result[i * n + j] += l[i * n + k] * u[k * n + j];
        }
}

double compute_error(double* original, double* lu_product, int n) {
    double error = 0.0;
    for (int i = 0; i < n * n; i++)
        error += fabs(original[i] - lu_product[i]);
    return error;
}

void restore_column_order(double* matrix, int* col_perm, int n) {
    int* inv_perm = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
        inv_perm[col_perm[i]] = i;

    double* temp = (double*)malloc(n * n * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            temp[i * n + j] = matrix[i * n + inv_perm[j]];

    for (int i = 0; i < n * n; i++)
        matrix[i] = temp[i];

    free(inv_perm);
    free(temp);
}

int main(int argc, char* argv[]) {
    int myrank, nprocs;
    int n = 1 << atoi(argv[1]);;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    double* a = NULL, * l = NULL, * u = NULL;
    double* original = NULL;

    if (myrank == 0) {
        a = (double*)malloc(n * n * sizeof(double));
        l = (double*)calloc(n * n, sizeof(double));
        u = (double*)calloc(n * n, sizeof(double));
        original = (double*)malloc(n * n * sizeof(double));

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                a[i * n + j] = hilbert_value(i, j);
                original[i * n + j] = a[i * n + j];
            }
    }

    if (myrank != 0) {
        a = (double*)malloc(n * n * sizeof(double));
        l = (double*)calloc(n * n, sizeof(double));
        u = (double*)calloc(n * n, sizeof(double));
    }
    MPI_Bcast(a, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double start = MPI_Wtime();

    // -----------------------------------------------
    //          Simple LU-decomposition 
    // -----------------------------------------------
    //for (int k = 0; k < n; k++) {
    //    if (k % nprocs == myrank) {
    //        for (int j = k; j < n; j++) {
    //            double sum = 0.0;
    //            for (int s = 0; s < k; s++)
    //                sum += l[k * n + s] * u[s * n + j];
    //            u[k * n + j] = a[k * n + j] - sum;
    //        }

    //        for (int j = 0; j < k; j++)
    //            l[k * n + j] = 0.0;
    //        l[k * n + k] = 1.0;
    //    }

    //    MPI_Bcast(&u[k * n], n, MPI_DOUBLE, k % nprocs, MPI_COMM_WORLD);

    //    for (int i = k + 1; i < n; i++) {
    //        if (i % nprocs == myrank) {
    //            double sum = 0.0;
    //            for (int s = 0; s < k; s++)
    //                sum += l[i * n + s] * u[s * n + k];
    //            l[i * n + k] = (a[i * n + k] - sum) / u[k * n + k];

    //            for (int j = 0; j < k; j++)
    //                u[i * n + j] = 0.0;
    //        }
    //    }
    //    for (int i = k + 1; i < n; i++)
    //        MPI_Bcast(&l[i * n + k], 1, MPI_DOUBLE, i % nprocs, MPI_COMM_WORLD);
    //}


    // ------------------------------------------------------
    //          LU-decomposition with main by rows
    // ------------------------------------------------------
    //int* col_perm = (int*)malloc(n * sizeof(int));
    //for (int i = 0; i < n; i++) col_perm[i] = i;

    //for (int k = 0; k < n; k++) {
    //    // Выбор максимального элемента по строке
    //    int max_col = k;
    //    double max_val = fabs(a[k * n + k]);
    //    for (int j = k + 1; j < n; j++) {
    //        if (fabs(a[k * n + j]) > max_val) {
    //            max_val = fabs(a[k * n + j]);
    //            max_col = j;
    //        }
    //    }

    //    // Меняем столбцы местами в строке k
    //    if (max_col != k) {
    //        for (int i = 0; i < n; i++) {
    //            double tmp = a[i * n + k];
    //            a[i * n + k] = a[i * n + max_col];
    //            a[i * n + max_col] = tmp;
    //        }

    //        // Обновляем порядок столбцов
    //        int tmp_idx = col_perm[k];
    //        col_perm[k] = col_perm[max_col];
    //        col_perm[max_col] = tmp_idx;
    //    }

    //    MPI_Barrier(MPI_COMM_WORLD);  // синхронизация всех процессов

    //    // Остальное — как в обычном LU (без выбора)
    //    if (k % nprocs == myrank) {
    //        for (int j = k; j < n; j++) {
    //            double sum = 0.0;
    //            for (int s = 0; s < k; s++)
    //                sum += l[k * n + s] * u[s * n + j];
    //            u[k * n + j] = a[k * n + j] - sum;
    //        }
    //        l[k * n + k] = 1.0;
    //    }

    //    MPI_Bcast(&u[k * n], n, MPI_DOUBLE, k % nprocs, MPI_COMM_WORLD);

    //    for (int i = k + 1; i < n; i++) {
    //        if (i % nprocs == myrank) {
    //            double sum = 0.0;
    //            for (int s = 0; s < k; s++)
    //                sum += l[i * n + s] * u[s * n + k];
    //            l[i * n + k] = (a[i * n + k] - sum) / u[k * n + k];
    //        }
    //    }
    //    for (int i = k + 1; i < n; i++)
    //        MPI_Bcast(&l[i * n + k], 1, MPI_DOUBLE, i % nprocs, MPI_COMM_WORLD);
    //}

    // ------------------------------------------------------
    //          LU-decomposition with main by cols
    // ------------------------------------------------------
    //for (int k = 0; k < n; k++) {
    //    // Ищем ведущую строку в столбце k (от k до n-1)
    //    double max_val = 0.0;
    //    int max_row = -1;
    //    if (myrank == 0) {
    //        max_val = fabs(a[k * n + k]);
    //        max_row = k;
    //        for (int i = k + 1; i < n; i++) {
    //            if (fabs(a[i * n + k]) > max_val) {
    //                max_val = fabs(a[i * n + k]);
    //                max_row = i;
    //            }
    //        }
    //    }

    //    // Широковещаем выбранную строку всем
    //    MPI_Bcast(&max_row, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //    if (max_row != k) {
    //        // Обмен строк между процессами
    //        if ((k % nprocs) == myrank || (max_row % nprocs) == myrank) {
    //            double* row_k = &a[k * n];
    //            double* row_m = &a[max_row * n];

    //            double* temp = (double*)malloc(n * sizeof(double));
    //            if (myrank == k % nprocs)
    //                MPI_Sendrecv(row_k, n, MPI_DOUBLE, max_row % nprocs, 0,
    //                    temp, n, MPI_DOUBLE, max_row % nprocs, 0,
    //                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //            if (myrank == max_row % nprocs)
    //                MPI_Sendrecv(row_m, n, MPI_DOUBLE, k % nprocs, 0,
    //                    temp, n, MPI_DOUBLE, k % nprocs, 0,
    //                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    //            if (myrank == k % nprocs)
    //                for (int j = 0; j < n; j++) row_k[j] = temp[j];
    //            if (myrank == max_row % nprocs)
    //                for (int j = 0; j < n; j++) row_m[j] = temp[j];
    //            free(temp);
    //        }
    //    }

    //    MPI_Barrier(MPI_COMM_WORLD);  // синхронизация после обмена

    //    if (k % nprocs == myrank) {
    //        for (int j = k; j < n; j++) {
    //            double sum = 0.0;
    //            for (int s = 0; s < k; s++)
    //                sum += l[k * n + s] * u[s * n + j];
    //            u[k * n + j] = a[k * n + j] - sum;
    //        }
    //        l[k * n + k] = 1.0;
    //    }

    //    MPI_Bcast(&u[k * n], n, MPI_DOUBLE, k % nprocs, MPI_COMM_WORLD);

    //    for (int i = k + 1; i < n; i++) {
    //        if (i % nprocs == myrank) {
    //            double sum = 0.0;
    //            for (int s = 0; s < k; s++)
    //                sum += l[i * n + s] * u[s * n + k];
    //            l[i * n + k] = (a[i * n + k] - sum) / u[k * n + k];
    //        }
    //    }

    //    for (int i = k + 1; i < n; i++)
    //        MPI_Bcast(&l[i * n + k], 1, MPI_DOUBLE, i % nprocs, MPI_COMM_WORLD);
    //}

    // ------------------------------------------------------
    //          LU-decomposition with main by rows and cols
    // ------------------------------------------------------
    int* col_perm = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) col_perm[i] = i;

    for (int k = 0; k < n; k++) {
        int pivot_row = k, pivot_col = k;
        double max_val = 0.0;

        // Процесс 0 ищет максимум в подматрице
        if (myrank == 0) {
            for (int i = k; i < n; i++) {
                for (int j = k; j < n; j++) {
                    double val = fabs(a[i * n + j]);
                    if (val > max_val) {
                        max_val = val;
                        pivot_row = i;
                        pivot_col = j;
                    }
                }
            }
        }

        // Рассылаем координаты главного элемента
        int pivot_info[2];
        if (myrank == 0) {
            pivot_info[0] = pivot_row;
            pivot_info[1] = pivot_col;
        }
        MPI_Bcast(pivot_info, 2, MPI_INT, 0, MPI_COMM_WORLD);
        pivot_row = pivot_info[0];
        pivot_col = pivot_info[1];

        // Меняем строки
        if ((k % nprocs) == myrank || (pivot_row % nprocs) == myrank) {
            double* temp = (double*)malloc(n * sizeof(double));
            if (myrank == k % nprocs)
                MPI_Sendrecv(&a[k * n], n, MPI_DOUBLE, pivot_row % nprocs, 0,
                    temp, n, MPI_DOUBLE, pivot_row % nprocs, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (myrank == pivot_row % nprocs)
                MPI_Sendrecv(&a[pivot_row * n], n, MPI_DOUBLE, k % nprocs, 0,
                    temp, n, MPI_DOUBLE, k % nprocs, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (myrank == k % nprocs)
                for (int j = 0; j < n; j++) a[k * n + j] = temp[j];
            if (myrank == pivot_row % nprocs)
                for (int j = 0; j < n; j++) a[pivot_row * n + j] = temp[j];
            free(temp);
        }

        // Меняем столбцы во всех строках
        for (int i = 0; i < n; i++) {
            double tmp = a[i * n + k];
            a[i * n + k] = a[i * n + pivot_col];
            a[i * n + pivot_col] = tmp;
        }

        int tmp = col_perm[k];
        col_perm[k] = col_perm[pivot_col];
        col_perm[pivot_col] = tmp;

        MPI_Barrier(MPI_COMM_WORLD);

        if (k % nprocs == myrank) {
            for (int j = k; j < n; j++) {
                double sum = 0.0;
                for (int s = 0; s < k; s++)
                    sum += l[k * n + s] * u[s * n + j];
                u[k * n + j] = a[k * n + j] - sum;
            }
            l[k * n + k] = 1.0;
        }

        MPI_Bcast(&u[k * n], n, MPI_DOUBLE, k % nprocs, MPI_COMM_WORLD);

        for (int i = k + 1; i < n; i++) {
            if (i % nprocs == myrank) {
                double sum = 0.0;
                for (int s = 0; s < k; s++)
                    sum += l[i * n + s] * u[s * n + k];
                l[i * n + k] = (a[i * n + k] - sum) / u[k * n + k];
            }
        }

        for (int i = k + 1; i < n; i++)
            MPI_Bcast(&l[i * n + k], 1, MPI_DOUBLE, i % nprocs, MPI_COMM_WORLD);
    }


    double end = MPI_Wtime();

    if (myrank == 0) {
        double* result = (double*)calloc(n * n, sizeof(double));
        multiply_lu(l, u, result, n);
        restore_column_order(result, col_perm, n);
        double error = compute_error(original, result, n);

        printf("Matrix size: %d\n", n);
        printf("Execution time: %lf sec\n", end - start);
        printf("Accuracy (mean of absolute deviations): %lf\n", error / n);

        free(original);
        free(result);
    }

    free(a);
    free(l);
    free(u);
    free(col_perm);

    MPI_Finalize();
    return 0;
}
