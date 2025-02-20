__kernel void vecAdd(
    __global float* x,
    __global float* y,
    __global float* z,
    __global float* w,
    int n,
    float a,
    float b
)
{                               
    int id = get_global_id(0);
                  
    if (id < n)
        w[id] = a * x[id] + b * y[id] * z[id];
}
