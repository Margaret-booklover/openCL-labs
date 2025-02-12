#include "myOpencl.h"

int main()
{
	const int power = 10;
	const int g_cuNumItems = 1 << power;
	const int a = 6;
	const int b = 2;

	size_t bytes = g_cuNumItems * sizeof(int);
	int* h_x;
	int* h_y;
	int* h_z;
	int* h_w;
	int* h_w_CPU;
	srand(time(NULL));

	h_x = (int*)malloc(bytes);
	h_y = (int*)malloc(bytes);
	h_z = (int*)malloc(bytes);
	h_w = (int*)malloc(bytes);
	h_w_CPU = (int*)malloc(bytes);

	for (int i = 0; i < g_cuNumItems; i++)
	{
		h_x[i] = rand() % 101;
		h_y[i] = rand() % 101;
		h_z[i] = rand() % 101;
	}

	cl_device_id deviceID = getDeviceInfo();
	cout << "size is " << g_cuNumItems << endl;
	
	// 5. Создание контекста
	cl_context context = createContext(deviceID);
	
	// 6. Создание очереди команд
	cl_command_queue queue = createQueue(deviceID, context);
	
	// 7. Создание программы
	// 8. Сборка программы
	cl_program program = build_program(context, deviceID, PROGRAM_FILE);
	
	// 9. Получение ядра
	cl_kernel kernel = createKernel(program, "vecAdd");
	
	// 10. Создание буфера
	cl_int errcode_ret = CL_SUCCESS;
	// Device input buffers
	cl_mem d_x, d_y, d_z;
	// Device output buffer
	cl_mem d_w;
	d_x = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
	d_y = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
	d_z = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
	d_w = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
	if (errcode_ret != CL_SUCCESS)
	{
		printf("Error to create buffer");
		return 0;
	}
	
	// 11. Установка буфера в качестве аргумента ядра
	cl_int err = clEnqueueWriteBuffer(queue, d_x, CL_TRUE, 0, bytes, h_x, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_y, CL_TRUE, 0, bytes, h_y, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_z, CL_TRUE, 0, bytes, h_z, 0, NULL, NULL);

	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_x);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_y);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_z);
	err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_w);
	err |= clSetKernelArg(kernel, 4, sizeof(unsigned int), &g_cuNumItems);
	err |= clSetKernelArg(kernel, 5, sizeof(unsigned int), &a);
	err |= clSetKernelArg(kernel, 6, sizeof(unsigned int), &b);
	if (err != CL_SUCCESS)
	{
		printf("Error to set kernel arg");
		return 0;
	}
	
	// 12. Запуск ядра
	cl_event event = executeKernel(g_cuNumItems, queue, kernel);
	
	// 13. Отображение буфера в память управляющего узла
	errcode_ret = CL_SUCCESS;
	cl_int puData = clEnqueueReadBuffer(queue, d_w, CL_TRUE, 0, bytes, h_w, 0, NULL, NULL);
	if (errcode_ret != CL_SUCCESS)
	{
		printf("Error to create context");
		return 0;
	}

	// 14. Использование результатов
	cl_ulong time_start, time_end;
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	double nanoSeconds = time_end - time_start;
	printf("OpenCl Execution time is: %0.3f milliseconds \n", nanoSeconds / 1000000.0);
	auto start = chrono::high_resolution_clock::now();

	for (int i = 0; i < g_cuNumItems; i++)
	{
		h_w_CPU[i] = a * h_x[i] + b * h_y[i] * h_z[i];
	}
	auto finish = chrono::high_resolution_clock::now();
	cout << "CPU Execution time is: " << (double)chrono::duration_cast<chrono::nanoseconds>(finish - start).count() << " nanoseconds\n";

	for (int i = 0; i < g_cuNumItems; i++)
	{
		if (h_w[i] != h_w_CPU[i])
		{
			cout << "index " << i << ", expected: " << h_w_CPU[i] << ", got " << h_w[i] << endl;
			break;
		}
	}
	
	// 15. Завершение отображения буфера
	clEnqueueUnmapMemObject(queue, d_w, h_w, 0, NULL, NULL);
	
	// 16. Удаление объектов и освобождение памяти управляющего узла
	clReleaseMemObject(d_x);
	clReleaseMemObject(d_y);
	clReleaseMemObject(d_z);
	clReleaseMemObject(d_w);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	free(h_x);
	free(h_y);
	free(h_z);
	free(h_w);
	return  0;
}
