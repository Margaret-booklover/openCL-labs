#include <stdio.h>
#include <mpi.h>
#define _USE_MATH_DEFINES
#include <math.h>

double f(double x)
{
	return 1 / (1 + x * x);
}

int lab4(int argc, char* argv[])
{
	double pi, sum, sum_odd, sum_even, term, h, error, start_time, end_time;
	int myrank, nprocs, n, i, j;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (myrank == 0)
    {
        printf("Running computations for different values of n...\n");
        printf("---------------------------------------------------\n");
        printf("i\tComputed value of pi\tError\tTime, ms\n");
    }

    for (j = 9; j <= 30; j++)
    {
        n = 1 << j;
        sum = 0;
        sum_odd = 0;
        sum_even = 0;

        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();

        h = 1.0 / n;

        // средних прямоугольников
        for (i = myrank + 1; i <= n; i += nprocs)
            sum += f(h * (i - 0.5));
        term = 4 * h * sum;

        // трапеций
        //for (i = myrank + 1; i <= n - 1; i += nprocs)
        //    sum += f(i * h);
        //term = 4 * h * (0.5 * (f(0) + f(1)) + sum);

        // Симпсона
        //for (i = myrank + 1; i <= n - 1; i += nprocs)
        //{
        //    if (i % 2 == 1)
        //        sum_odd += f(i * h); // Нечётные индексы
        //    else
        //        sum_even += f(i * h); // Чётные индексы
        //}
        //term = 4 * h / 3.0 * (f(0) + 4 * sum_odd + 2 * sum_even + f(1));

        MPI_Reduce(&term, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        end_time = MPI_Wtime();

        if (myrank == 0)
        {
            error = fabs(pi - M_PI);
            printf("%d | %.15f | %.15f | %.6f\n", j, pi, error, (end_time - start_time) * 1000);
        }
    }

    MPI_Finalize();
	return 0;
}