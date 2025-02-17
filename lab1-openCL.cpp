#include "myOpencl.h"

int main()
{
	const int power = 2;
	const int g_cuNumItems = 1 << power;
	const int a = 6;
	const int b = 2;
	const int M = g_cuNumItems;
	const int K = g_cuNumItems;
	const int N = 1;

	size_t bytes = g_cuNumItems * sizeof(int);
	size_t bytes2 = g_cuNumItems * g_cuNumItems * sizeof(int);
	int maxRand = 2;
	int* h_x;
	int* h_y;
	int* h_Y;
	int* h_z;
	int* h_w;
	int* h_W;
	int* h_w_CPU;
	int* h_W_CPU;
	srand(time(NULL));

	h_x = (int*)malloc(bytes);
	h_y = (int*)malloc(bytes);
	h_Y = (int*)malloc(bytes2);
	h_z = (int*)malloc(bytes);
	h_w = (int*)malloc(bytes);
	h_W = (int*)malloc(bytes);
	h_w_CPU = (int*)malloc(bytes);
	h_W_CPU = (int*)malloc(bytes);

	for (int i = 0; i < g_cuNumItems; i++)
	{
		h_x[i] = rand() % maxRand;
		h_y[i] = rand() % maxRand;
		h_z[i] = rand() % maxRand;
		for (int k = 0; k < g_cuNumItems; k++)
		{
			h_Y[i*g_cuNumItems + k] = rand() % maxRand;
		}
	}

	cl_device_id deviceID = getDeviceInfo();
	cout << "size is " << g_cuNumItems << endl;
	
	// 5. Создание контекста
	cl_context context = createContext(deviceID);
	cl_context context2 = createContext(deviceID);
	
	// 6. Создание очереди команд
	cl_command_queue queue = createQueue(deviceID, context);
	cl_command_queue queue2 = createQueue(deviceID, context2);
	
	// 7. Создание программы
	// 8. Сборка программы
	cl_program program = build_program(context, deviceID, PROGRAM_FILE);
	cl_program program2 = build_program(context2, deviceID, PROGRAM_FILE2);
	
	// 9. Получение ядра
	cl_kernel kernel = createKernel(program, "vecAdd");
	cl_kernel kernel2 = createKernel(program2, "matMul");
	
	// 10. Создание буфера
	cl_int errcode_ret = CL_SUCCESS;
	// Device input buffers
	cl_mem d_x, d_X, d_y, d_z, d_Y;
	// Device output buffer
	cl_mem d_w, d_W;
	d_x = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
	d_X = clCreateBuffer(context2, CL_MEM_READ_ONLY, bytes, NULL, NULL);
	d_y = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
	d_Y = clCreateBuffer(context2, CL_MEM_READ_ONLY, bytes2, NULL, NULL);
	d_z = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
	d_w = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
	d_W = clCreateBuffer(context2, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
	if (errcode_ret != CL_SUCCESS)
	{
		printf("Error to create buffer");
		return 0;
	}
	
	// 11. Установка буфера в качестве аргумента ядра
	cl_int err = clEnqueueWriteBuffer(queue, d_x, CL_TRUE, 0, bytes, h_x, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue2, d_X, CL_TRUE, 0, bytes, h_x, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_y, CL_TRUE, 0, bytes, h_y, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue2, d_Y, CL_TRUE, 0, bytes2, h_Y, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue2, d_W, CL_TRUE, 0, bytes2, h_W, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_z, CL_TRUE, 0, bytes, h_z, 0, NULL, NULL);

	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_x);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_y);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_z);
	err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_w);
	err |= clSetKernelArg(kernel, 4, sizeof(unsigned int), &g_cuNumItems);
	err |= clSetKernelArg(kernel, 5, sizeof(unsigned int), &a);
	err |= clSetKernelArg(kernel, 6, sizeof(unsigned int), &b);

	err |= clSetKernelArg(kernel2, 0, sizeof(int), &M);
	err |= clSetKernelArg(kernel2, 1, sizeof(int), &a);
	err |= clSetKernelArg(kernel2, 2, sizeof(cl_mem), &d_X);
	err |= clSetKernelArg(kernel2, 3, sizeof(cl_mem), &d_Y);
	err |= clSetKernelArg(kernel2, 4, sizeof(cl_mem), &d_W);
	if (errcode_ret != CL_SUCCESS)
	{
		printf("Error to set kernel args");
		return 0;
	}
	cout << "Args set" << endl;
	
	// 12. Запуск ядра
	size_t s = { g_cuNumItems };
	cl_event event = executeKernel(&s, NULL, queue, kernel, 1);
	cout << "Kernel1 done" << endl;

	//const int TS = 16;
	//size_t local[2] = { TS, N };
	//size_t global[2] = { M, N };
	cl_event event2 = executeKernel(&s, NULL, queue2, kernel2, 2);
	cout << "Kernel2 done" << endl;
	
	// 13. Отображение буфера в память управляющего узла
	errcode_ret = CL_SUCCESS;
	cl_int puData = clEnqueueReadBuffer(queue, d_w, CL_TRUE, 0, bytes, h_w, 0, NULL, NULL);
	cl_int puData2 = clEnqueueReadBuffer(queue2, d_W, CL_TRUE, 0, bytes, h_W, 0, NULL, NULL);
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

	cout << "Vector addition on CPU" << endl;
	auto start = chrono::high_resolution_clock::now();

	for (int i = 0; i < g_cuNumItems; i++)
	{
		h_w_CPU[i] = a * h_x[i] + b * h_y[i] * h_z[i];
	}
	auto finish = chrono::high_resolution_clock::now();
	cout << "Checking results... ";
	bool flag = true;
	for (int i = 0; i < g_cuNumItems; i++)
	{
		if (h_w[i] != h_w_CPU[i])
		{
			cout << "index " << i << ", expected: " << h_w_CPU[i] << ", got " << h_w[i] << endl;
			flag = false;
			break;
		}
	}
	if (flag) cout << "OK";
	else cout << "ERROR";
	cout << endl << "CPU Execution time is: " << (double)chrono::duration_cast<chrono::nanoseconds>(finish - start).count() << " nanoseconds\n" << endl;

	cout << "Matrix to vector multiplication on CPU" << endl;

	clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	nanoSeconds = time_end - time_start;
	printf("OpenCl Execution time is: %0.3f milliseconds \n", nanoSeconds / 1000000.0);

	start = chrono::high_resolution_clock::now();

	for (int m = 0; m < g_cuNumItems; m++) 
	{
		int acc = 0;
		for (int k = 0; k < g_cuNumItems; k++)
		{
			cout << h_Y[k * g_cuNumItems + m] << "\t";
			acc += h_Y[k * g_cuNumItems + m] * h_x[k];
		}
		h_W_CPU[m] = a * acc;
		cout << endl;
	}
	finish = chrono::high_resolution_clock::now();
	cout << "Checking results... ";
	flag = true;
	for (int i = 0; i < g_cuNumItems; i++)
	{
		cout << h_x[i] << "\t";
		/*if (h_W[i] != h_W_CPU[i])
		{
			cout << "index " << i << ", expected: " << h_W_CPU[i] << ", got " << h_W[i] << endl;
			flag = false;
			break;
		}*/
	}
	cout << endl;
	for (int i = 0; i < g_cuNumItems; i++)
	{
		cout << h_W[i] << "\t" << h_W_CPU[i] << endl;
	}
	if (flag) cout << "OK";
	else cout << "ERROR";
	cout << endl << "CPU Execution time is: " << (double)chrono::duration_cast<chrono::nanoseconds>(finish - start).count() << " nanoseconds\n";

	
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
