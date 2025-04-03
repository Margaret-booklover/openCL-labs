__kernel void boxBlur(
    __read_only image2d_t imgSrc,
    __constant float * kernelValues,
    __global int * kernelSize,
    __write_only image2d_t imgConvolved)
{
    const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

    int w = kernelSize[0];
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Локальная память для хранения части изображения
    __local float4 localData[16][16]; // Размер зависит от размера рабочей группы

    // Загрузка данных в локальную память
    int localX = get_local_id(0);
    int localY = get_local_id(1);
    int groupIdX = get_group_id(0);
    int groupIdY = get_group_id(1);

    int globalX = groupIdX * get_local_size(0) + localX;
    int globalY = groupIdY * get_local_size(1) + localY;

    if (globalX < get_image_width(imgSrc) && globalY < get_image_height(imgSrc)) {
        int2 coords = (int2)(globalX, globalY);
        uint4 pix = read_imageui(imgSrc, smp, coords);
        localData[localY][localX] = (float4)((float)pix.x, (float)pix.y, (float)pix.z, (float)pix.w);
    }

    // Синхронизация рабочих элементов
    barrier(CLK_LOCAL_MEM_FENCE);

    // Выполнение свертки
    float4 convPix = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < w; j++) {
            int2 coords = (int2)(x - (w / 2) + i, y - (w / 2) + j);
            if (coords.x >= 0 && coords.x < get_image_width(imgSrc) &&
                coords.y >= 0 && coords.y < get_image_height(imgSrc)) {
                convPix += localData[localY + j][localX + i] * kernelValues[i + w * j];
            }
        }
    }

    // Запись результата
    int2 outputCoords = (int2)(x, y);
    uint4 result = (uint4)((uint)convPix.x, (uint)convPix.y, (uint)convPix.z, (uint)convPix.w);
    write_imageui(imgConvolved, outputCoords, result);
}