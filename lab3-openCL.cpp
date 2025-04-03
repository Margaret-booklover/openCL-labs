#include "myOpencl.h"
#include <cmath>

template <typename T>
Mat createMat(T* data, int rows, int cols, int chs = 1) {
    Mat mat(rows, cols, CV_MAKETYPE(DataType<T>::type, chs));
    memcpy(mat.data, data, rows * cols * chs * sizeof(T));
    return mat;
}

//void generateBoxBlurKernel(float* kernel, int kernelSize) {
//    float weight = 1.0f / (kernelSize * kernelSize);
//    for (int i = 0; i < kernelSize * kernelSize; ++i) {
//        kernel[i] = weight;
//    }
//}

//void generateGaussianKernel(float* kernel, int kernelSize, float sigma) {
//    float sum = 0.0f;
//    int radius = kernelSize / 2;
//    for (int y = -radius; y <= radius; ++y) {
//        for (int x = -radius; x <= radius; ++x) {
//            int index = (y + radius) * kernelSize + (x + radius);
//            kernel[index] = exp(-(x * x + y * y) / (2 * sigma * sigma));
//            sum += kernel[index];
//        }
//    }
//    // Нормализация ядра
//    for (int i = 0; i < kernelSize * kernelSize; ++i) {
//        kernel[i] /= sum;
//    }
//}

int lab3()
{
    cout << "Lab3: Box Blur using Global Memory" << endl;
    const char* filename = "forest.bmp";

    // 1. Получение информации об устройстве
    cl_device_id deviceID = getDeviceInfo();

    // 2. Создание контекста
    cl_context context = createContext(deviceID);

    // 3. Создание очереди команд
    cl_command_queue queue = createQueue(deviceID, context, 2);

    // 4. Создание программы
    cl_int errcode_ret = CL_SUCCESS;
    cl_program program = build_program(context, deviceID, CONVOLUTION_PROGRAM);

    // 5. Получение ядра
    cl_kernel kernel = createKernel(program, "boxBlur");

    cl_mem image1, image2;
    Mat image = createRGBAImages(filename, &image1, &image2, context);

    // 11. Подготовка ядра фильтра
    const int kernelSize = 10; // Размер ядра
    float* kernelValues = new float[kernelSize * kernelSize];

    float value = 1.0f / (kernelSize * kernelSize);
    for (int i = 0; i < kernelSize * kernelSize; ++i) {
        kernelValues[i] = value;
    }

    cl_mem kernelValuesBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * kernelSize * kernelSize, kernelValues, &errcode_ret);
    if (errcode_ret != CL_SUCCESS) {
        printf("Error creating kernel values buffer\n");
        return 0;
    }

    cl_mem kernelSizeBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), (void*)&kernelSize, &errcode_ret);
    if (errcode_ret != CL_SUCCESS) {
        printf("Error creating kernel size buffer\n");
        return 0;
    }

    // 12. Установка аргументов ядра
    errcode_ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&image1);
    errcode_ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&kernelValuesBuffer);
    errcode_ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&kernelSizeBuffer);
    errcode_ret |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&image2);

    if (errcode_ret != CL_SUCCESS) {
        printf("Error setting kernel arguments: %d\n", errcode_ret);
        return 0;
    }

    const size_t global_work_size[] = { image.cols, image.rows };

    cl_event event;
    errcode_ret = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, &event);
    if (errcode_ret != CL_SUCCESS) {
        printf("Error enqueuing kernel: %d\n", errcode_ret);
        return 0;
    }

    clWaitForEvents(1, &event);
    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double nanoSeconds = time_end - time_start;
    printf("OpenCl Execution time is : % 0.3f milliseconds \n", nanoSeconds / 1000000.0);

    clReleaseEvent(event);

    // 14. Чтение результата
    size_t origin[] = { 0, 0, 0 };
    size_t region[] = { image.cols, image.rows, 1 };
    unsigned char* buffer_result = (unsigned char*)calloc(image.cols * image.rows * 4, sizeof(unsigned char));
    errcode_ret = clEnqueueReadImage(queue, image2, CL_TRUE, origin, region, 0, 0, buffer_result, 0, NULL, NULL);
    if (errcode_ret != CL_SUCCESS) {
        printf("Error reading result image: %d\n", errcode_ret);
        return 0;
    }

    // 15. Преобразование результата в cv::Mat
    Mat image_result = createMat<unsigned char>(buffer_result, image.rows, image.cols, 4);

    // 16. Конвертация из RGBA в BGR для отображения
    cvtColor(image_result, image_result, COLOR_RGBA2BGR);

    // 17. Отображение результата
    namedWindow("Result", WINDOW_AUTOSIZE);
    imshow("Result", image_result);
    waitKey(0);

    // 18. Освобождение ресурсов
    clReleaseMemObject(kernelValuesBuffer);
    clReleaseMemObject(kernelSizeBuffer);
    clReleaseMemObject(image1);
    clReleaseMemObject(image2);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(buffer_result);
    delete[] kernelValues;

    return 0;
}