__kernel void vecAdd(
    __global int* x,
    __global int* y,
    __global int* z,
    __global int* w,
    const unsigned int n,
    const int a,
    const int b
)
{                               
    int id = get_global_id(0);
                  
    if (id < n)
        w[id] = a * x[id] + b * y[id] * z[id];
}
