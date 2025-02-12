__kernel void matMul(
    const int M, 
    const int N, 
    const int K,
    const int a,
    __global int* A,
    __global int* B,
    __global int* C
) 
{
    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)

    // Compute a single element (loop over K)
    int acc = 0;
    for (int k = 0; k < K; k++) 
    {
        acc += A[k * M + globalRow] * B[globalCol * K + k];
    }

    // Store the result
    C[globalCol * M + globalRow] = a * acc;
}