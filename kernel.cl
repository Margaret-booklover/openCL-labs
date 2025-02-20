__kernel void vecAdd(
    __global double* x,
    __global double* y,
    __global double* z,
    __global double* w,
    int n,
    double a,
    double b
)
{                               
    int id = get_global_id(0);
                  
    if (id < n)
        w[id] = a * x[id] + b * y[id] * z[id];
}
