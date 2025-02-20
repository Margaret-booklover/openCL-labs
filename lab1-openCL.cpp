#include "myOpencl.h"
#include <vector>

int runProgram(size_t power, bool execKernel2)
{
	int i, k, m;
	size_t N = 1 << power;
	const float a = 6;
	const float b = 2;

	size_t bytes = N * sizeof(float);
	size_t bytes2 = N * N * sizeof(float);
	//float maxRand = 2;
	float* h_x;
	float* h_y;
	float* h_z;
	float* h_w;
	float* h_w_CPU;
	srand(time(NULL));

	h_x = (float*)malloc(bytes);
	h_y = (float*)malloc(bytes);
	h_z = (float*)malloc(bytes);
	h_w = (float*)malloc(bytes);
	h_w_CPU = (float*)malloc(bytes);

	for (i = 0; i < N; i++)
	{
		h_x[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		h_y[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		h_z[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	}
	cout << "Vectors initialized" << endl;

	cl_device_id deviceID = getDeviceInfo();
	cout << "power is " << power << ", size is " << N << endl;
	
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
	cl_float errcode_ret = CL_SUCCESS;
	// Device input buffers
	cl_mem d_x, d_X, d_y, d_z, d_Y;
	// Device output buffer
	cl_mem d_w, d_W;
	d_x = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
	d_y = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
	d_z = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
	d_w = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
	if (errcode_ret != CL_SUCCESS)
	{
		cout << "Error to create buffer" << endl;
		return 0;
	}
	cout << "Buffer created" << endl;
	
	// 11. Установка буфера в качестве аргумента ядра
	cl_int err = clEnqueueWriteBuffer(queue, d_x, CL_TRUE, 0, bytes, h_x, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_y, CL_TRUE, 0, bytes, h_y, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_z, CL_TRUE, 0, bytes, h_z, 0, NULL, NULL);

	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_x);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_y);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_z);
	err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_w);
	err |= clSetKernelArg(kernel, 4, sizeof(int), &N);
	err |= clSetKernelArg(kernel, 5, sizeof(float), &a);
	err |= clSetKernelArg(kernel, 6, sizeof(float), &b);

	if (errcode_ret != CL_SUCCESS)
	{
		cout << "Error to set kernel args" << endl;
		return 0;
	}
	
	// 12. Запуск ядра
	size_t s = { N };
	cl_event event = executeKernel(&s, NULL, queue, kernel, 1);
	//cout << "Kernel1 done" << endl;
	
	// 13. Отображение буфера в память управляющего узла
	errcode_ret = CL_SUCCESS;
	cl_float puData = clEnqueueReadBuffer(queue, d_w, CL_TRUE, 0, bytes, h_w, 0, NULL, NULL);
	if (errcode_ret != CL_SUCCESS)
	{
		cout << "Error to read buffer" << endl;
		return 0;
	}

	// 14. Использование результатов
	cl_ulong time_start, time_end;
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	double nanoSeconds = time_end - time_start;
	printf("OpenCl Execution time is : % 0.3f milliseconds \n", nanoSeconds / 1000000.0);

	cout << "Vector addition on CPU" << endl;
	auto start = chrono::high_resolution_clock::now();

	for (i = 0; i < N; i++)
	{
		h_w_CPU[i] = a * h_x[i] + b * h_y[i] * h_z[i];
	}
	auto finish = chrono::high_resolution_clock::now();
	bool flag;
	//cout << "Checking results... ";
	//flag = true;
	//for (i = 0; i < N; i++)
	//{
	//	if (h_w[i] != h_w_CPU[i])
	//	{
	//		cout << "index " << i << ", expected: " << h_w_CPU[i] << ", got " << h_w[i] << endl;
	//		flag = false;
	//		break;
	//	}
	//}
	//if (flag) cout << "OK";
	//else cout << "ERROR";
	cout << endl << "CPU Execution time is: " << (double)chrono::duration_cast<chrono::nanoseconds>(finish - start).count() << " nanoseconds\n" << endl;

	if (execKernel2)
	{
		cout << "Matrix to vector multiplication on CPU" << endl;

		//float* h_Y;
		vector<float> h_Y(N * N);
		float* h_W;
		float* h_W_CPU;
		//h_Y = (float*)malloc(bytes2);
		h_W = (float*)malloc(bytes);
		h_W_CPU = (float*)malloc(bytes);

		for (i = 0; i < N; i++)
		{
			for (k = 0; k < N; k++)
			{
				h_Y[i * N + k] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
			}
		}

		cl_context context2 = createContext(deviceID);
		cl_command_queue queue2 = createQueue(deviceID, context2);
		cl_program program2 = build_program(context2, deviceID, PROGRAM_FILE2);
		cl_kernel kernel2 = createKernel(program2, "matMul");
		// Device input buffers
		cl_mem d_X, d_Y;
		// Device output buffer
		cl_mem d_W;
		d_X = clCreateBuffer(context2, CL_MEM_READ_ONLY, bytes, NULL, NULL);
		d_Y = clCreateBuffer(context2, CL_MEM_READ_ONLY, bytes2, NULL, NULL);
		d_W = clCreateBuffer(context2, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
		if (errcode_ret != CL_SUCCESS)
		{
			cout << "Error to create buffer" << endl;
			return 0;
		}
		cout << "Buffer created" << endl;

		// 11. Установка буфера в качестве аргумента ядра
		err = clEnqueueWriteBuffer(queue2, d_X, CL_TRUE, 0, bytes, h_x, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(queue2, d_Y, CL_TRUE, 0, bytes2, h_Y.data(), 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(queue2, d_W, CL_TRUE, 0, bytes2, h_W, 0, NULL, NULL);

		err |= clSetKernelArg(kernel2, 0, sizeof(int), &N);
		err |= clSetKernelArg(kernel2, 1, sizeof(float), &a);
		err |= clSetKernelArg(kernel2, 2, sizeof(cl_mem), &d_Y);
		err |= clSetKernelArg(kernel2, 3, sizeof(cl_mem), &d_X);
		err |= clSetKernelArg(kernel2, 4, sizeof(cl_mem), &d_W);

		if (errcode_ret != CL_SUCCESS)
		{
			cout << "Error to set kernel args" << endl;
			return 0;
		}

		cl_event event2 = executeKernel(&s, NULL, queue2, kernel2, 1);
		//cout << "Kernel2 done" << endl;

		// 13. Отображение буфера в память управляющего узла
		errcode_ret = CL_SUCCESS;
		cl_float puData2 = clEnqueueReadBuffer(queue2, d_W, CL_TRUE, 0, bytes, h_W, 0, NULL, NULL);
		if (errcode_ret != CL_SUCCESS)
		{
			cout << "Error to read buffer" << endl;
			return 0;
		}

		clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		nanoSeconds = time_end - time_start;
		printf("OpenCl Execution time is: %0.3f milliseconds \n", nanoSeconds / 1000000.0);

		start = chrono::high_resolution_clock::now();

		for (m = 0; m < N; m++)
		{
			float acc = 0;
			for (k = 0; k < N; k++)
			{
				acc += h_Y[m * N + k] * h_x[k];
			}
			h_W_CPU[m] = a * acc;
		}
		finish = chrono::high_resolution_clock::now();
		//cout << "Checking results... ";
		//flag = true;
		//for (i = 0; i < N; i++)
		//{
		//	if (h_W[i] != h_W_CPU[i])
		//	{
		//		cout << "index " << i << ", expected: " << h_W_CPU[i] << ", got " << h_W[i] << endl;
		//		flag = false;
		//		break;
		//	}
		//}
		//if (flag) cout << "OK";
		//else cout << "ERROR";
		cout << endl << "CPU Execution time is: " << (double)chrono::duration_cast<chrono::nanoseconds>(finish - start).count() << " nanoseconds\n";
		clReleaseKernel(kernel2);
		clReleaseProgram(program2);
		clReleaseCommandQueue(queue2);
		clReleaseContext(context2);
		clReleaseMemObject(d_Y);
		//free(h_Y);
		free(h_W);
		free(h_W_CPU);
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
	free(h_w_CPU);
	return  0;
}
float main()
{
	//const float power = 3;
	for (size_t i = 15; i < 21; i++)
	{
		runProgram(i, true);
		cout << endl << endl;
	}
}
