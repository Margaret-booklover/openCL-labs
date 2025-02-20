__kernel void matMul(
    int N, 
    float a,
    __global float* A,
    __global float* B,
    __global float* C
) 
{
    float sum = 0;
    int i = get_global_id(0); // row index
    for (int k = 0; k < N; k++)
    {
        sum += A[i * N + k] * B[k];
    }
    C[i] = a * sum;
}