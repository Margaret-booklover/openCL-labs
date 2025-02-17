__kernel void matMul(
    const int N, 
    const int a,
    __global int* A,
    __global int* B,
    __global int* C
) 
{
    int sum = 0;
    int i = get_global_id(0); // row index
    for (int k = 0; k < N; k++)
    {
        sum += A[i * N + k] * B[k];
    }
    C[i] = a * sum;
}