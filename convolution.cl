__kernel void boxBlur(__read_only image2d_t imgSrc,
                            __constant float * kernelValues,
                            __constant int * kernelSize,
                            __write_only image2d_t imgConvolved)
{
    const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates
                          CLK_ADDRESS_CLAMP | //Clamp to zeros
                          CLK_FILTER_NEAREST; //Don't interpolate

    int w = kernelSize[0];
    int x = get_global_id(0);
    int y = get_global_id(1);

    float4 convPix = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float4 temp;
    uint4 pix;
    int2 coords;

    for (int i = 0; i < w; i++)
    {
        for (int j = 0; j < w; j++)
        {
            coords.x = x - (w / 2) + i; // Смещение относительно центрального пикселя
            coords.y = y - (w / 2) + j;

            // Проверка границ
            if (coords.x >= 0 && coords.x < get_image_width(imgSrc) &&
                coords.y >= 0 && coords.y < get_image_height(imgSrc)) {
                pix = read_imageui(imgSrc, smp, coords);
                temp = (float4)((float)pix.x, (float)pix.y, (float)pix.z, (float)pix.w);
                convPix += temp * kernelValues[i + w * j];
            }
        }
    }

    coords.x = x + (w >> 1);
    coords.y = y + (w >> 1);
    pix = (uint4)((uint)convPix.x, (uint)convPix.y, (uint)convPix.z, (uint)convPix.w);
    write_imageui(imgConvolved, coords, pix);
}