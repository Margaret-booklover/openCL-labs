__kernel void matMul(
    int N, 
    double a,
    __global double* A,
    __global double* B,
    __global double* C
) 
{
    double sum = 0;
    int i = get_global_id(0); // row index
    for (int k = 0; k < N; k++)
    {
        sum += A[i * N + k] * B[k];
    }
    C[i] = a * sum;
}