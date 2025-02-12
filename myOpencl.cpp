#include "myOpencl.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

cl_device_id getDeviceInfo()
{
	cout << "-----------------------------------------" << endl;
	cout << "               DEVICE INFO" << endl;
	cout << "-----------------------------------------" << endl;
	// 1. Получение платформы
	cl_uint uNumPlatforms;
	clGetPlatformIDs(0, NULL, &uNumPlatforms);
	std::cout << uNumPlatforms << " platforms" << std::endl;
	cl_platform_id* pPlatforms = new cl_platform_id[uNumPlatforms];
	clGetPlatformIDs(uNumPlatforms, pPlatforms, &uNumPlatforms);

	// 2. Получение информации о платформе
	const size_t	size = 128;
	char			param_value[size] = { 0 };
	size_t 			param_value_size_ret = 0;
	for (int i = 0; i < uNumPlatforms; ++i)
	{
		cl_int res = clGetPlatformInfo(pPlatforms[i], CL_PLATFORM_NAME, size, static_cast<void*>(param_value), &param_value_size_ret);
		printf("Platform %i name is %s\n", pPlatforms[i], param_value);
		param_value_size_ret = 0;
	}

	// 3. Получение номера CL устройства
	int32_t		platform_id = 0;
	if (uNumPlatforms > 1)
	{
		platform_id = 0;
	}
	cl_device_id deviceID;
	cl_uint uNumGPU;
	clGetDeviceIDs(pPlatforms[platform_id], CL_DEVICE_TYPE_DEFAULT/*CL_DEVICE_TYPE_GPU*/, 1, &deviceID, &uNumGPU);

	// 4. Получение информации о CL устройстве
	param_value_size_ret = 0;
	cl_int res1 = clGetDeviceInfo(deviceID, CL_DEVICE_NAME, size, static_cast<void*>(param_value), &param_value_size_ret);
	printf("Device %i name is %s\n", deviceID, param_value);

	size_t res;
	res1 = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE, size, &res, &param_value_size_ret);
	printf("Max work group size is %i \n", res);

	delete[] pPlatforms;

	cout << "-----------------------------------------" << endl;

	return deviceID;
}

cl_command_queue createQueue(cl_device_id deviceID, cl_context context)
{
	cl_int errcode_ret = 0;
	cl_queue_properties qprop[] = { CL_QUEUE_PROPERTIES, (cl_command_queue_properties)CL_QUEUE_PROFILING_ENABLE, 0 };
	cl_command_queue queue = clCreateCommandQueueWithProperties(context, deviceID, qprop, &errcode_ret);
	if (errcode_ret != CL_SUCCESS)
	{
		switch (errcode_ret)
		{
		case CL_INVALID_CONTEXT:  printf("if context is not a valid context.\n");
			break;
		case CL_INVALID_DEVICE: printf("if device is not a valid device or is not associated with context.\n");
			break;
		case CL_INVALID_VALUE: printf("if values specified in properties are not valid.\n");
			break;
		case CL_INVALID_QUEUE_PROPERTIES: printf("if values specified in properties are valid but are not supported by the device.\n");
			break;
		case CL_OUT_OF_RESOURCES: printf("if there is a failure to allocate resources required by the OpenCL implementation on the device.\n");
			break;
		case CL_OUT_OF_HOST_MEMORY: printf("if there is a failure to allocate resources required by the OpenCL implementation on the host.\n");
			break;
		default:
			break;
		}
		printf("Error to create command queue");
		exit(1);
	}
	return queue;
}


cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

	cl_program program;
	FILE* program_handle;
	char* program_buffer, * program_log;
	size_t program_size, log_size;
	cl_int err;

	program_handle = fopen(filename, "rb");
	if (program_handle == NULL) {
		perror("Couldn't find the program file");
		exit(1);
	}
	fseek(program_handle, 0, SEEK_END);
	program_size = ftell(program_handle);
	rewind(program_handle);
	program_buffer = (char*)malloc(program_size + 1);
	program_buffer[program_size] = '\0';
	fread(program_buffer, sizeof(char), program_size, program_handle);
	fclose(program_handle);

	program = clCreateProgramWithSource(ctx, 1, (const char**)&program_buffer, &program_size, &err);
	if (err < 0) {
		perror("Couldn't create the program");
		exit(1);
	}
	free(program_buffer);
	err = clBuildProgram(program, 1, &dev, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		switch (err)
		{
		case CL_INVALID_PROGRAM:  printf(" if program is not a valid program object.\n");
			break;
		case CL_INVALID_VALUE:  printf(" if device_list is NULL and num_devices is greater than zero, or if device_list is not NULL and num_devices is zero.\n");
			break;
		case CL_INVALID_DEVICE:  printf(" if OpenCL devices listed in device_list are not in the list of devices associated with program.\n");
			break;
		case CL_INVALID_BINARY:  printf(" if program is created with clCreateWithProgramWithBinary and devices listed in device_list do not have a valid program binary loaded.\n");
			break;
		case CL_INVALID_BUILD_OPTIONS:  printf(" if the build options specified by options are invalid.\n");
			break;
		case CL_INVALID_OPERATION:  printf(" if the build of a program executable for any of the devices listed in device_list by a previous call to clBuildProgram for program has not completed.\n");
			break;
		case CL_COMPILER_NOT_AVAILABLE:  printf(" if program is created with clCreateProgramWithSource and a compiler is not available i.e.CL_DEVICE_COMPILER_AVAILABLE specified in the table of OpenCL Device Queries for clGetDeviceInfo is set to CL_FALSE.\n");
			break;
		case CL_BUILD_PROGRAM_FAILURE:  printf(" if there is a failure to build the program executable.This error will be returned if clBuildProgram does not return until the build has completed.\n");
			break;
		case CL_OUT_OF_HOST_MEMORY:  printf(" if there is a failure to allocate resources required by the OpenCL implementation on the host.\n");
			break;
		}

		if (err == CL_BUILD_PROGRAM_FAILURE) {
			// Determine the size of the log
			size_t log_size;
			clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

			// Allocate memory for the log
			char* log = (char*)malloc(log_size);

			// Get the log
			clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

			// Print the log
			printf("%s\n", log);
		}
		printf("Error to build program");
		exit(1);
	}

	return program;
}

cl_event executeKernel(size_t* uGlobalWorkSize, size_t* uLocalWorkSize, cl_command_queue queue, cl_kernel kernel, cl_uint work_dim)
{
	cl_int errcode_ret = CL_SUCCESS;
	cl_event event;
	errcode_ret = clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, uGlobalWorkSize, uLocalWorkSize, 0, NULL, &event);
	if (errcode_ret != CL_SUCCESS)
	{
		switch (errcode_ret)
		{
		case CL_INVALID_PROGRAM_EXECUTABLE:  printf("  if there is no successfully built program executable available for device associated with command_queue..\n");
			break;
		case CL_INVALID_COMMAND_QUEUE:  printf("  if command_queue is not a valid command - queue..\n");
			break;
		case CL_INVALID_KERNEL:  printf("  if kernel is not a valid kernel object..\n");
			break;
		case  CL_INVALID_CONTEXT:  printf("  if context associated with command_queue and kernel is not the same or if the context associated with command_queue and events in event_wait_list are not the same..\n");
			break;
		case CL_INVALID_KERNEL_ARGS:  printf("  if the kernel argument values have not been specified..\n");
			break;
		case CL_INVALID_WORK_DIMENSION:  printf(" if work_dim is not a valid value(i.e.a value between 1 and 3)..\n");
			break;
		case CL_INVALID_WORK_GROUP_SIZE:  printf(" if local_work_size is specified and number of work - items specified by global_work_size is not evenly divisable by size of work - group given by local_work_size or does not match the work - group size specified for kernel using the __attribute__((reqd_work_group_size(X, Y, Z))) qualifier in program source..\n");
			break;
		case CL_INVALID_WORK_ITEM_SIZE:  printf(" if the number of work - items specified in any of local_work_size[0], ... local_work_size[work_dim - 1] is greater than the corresponding values specified by CL_DEVICE_MAX_WORK_ITEM_SIZES[0], ....CL_DEVICE_MAX_WORK_ITEM_SIZES[work_dim - 1]..\n");
			break;
		case CL_INVALID_GLOBAL_OFFSET:  printf(" if global_work_offset is not NULL..\n");
			break;
		case CL_OUT_OF_RESOURCES:  printf(" if there is a failure to queue the execution instance of kernel on the command - queue because of insufficient resources needed to execute the kernel.For example, the explicitly specified local_work_size causes a failure to execute the kernel because of insufficient resources such as registers or local memory.Another example would be the number of read - only image args used in kernel exceed the CL_DEVICE_MAX_READ_IMAGE_ARGS value for device or the number of write - only image args used in kernel exceed the CL_DEVICE_MAX_WRITE_IMAGE_ARGS value for device or the number of samplers used in kernel exceed CL_DEVICE_MAX_SAMPLERS for device..\n");
			break;
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:  printf(" if there is a failure to allocate memory for data store associated with image or buffer objects specified as arguments to kernel..\n");
			break;
		case CL_INVALID_EVENT_WAIT_LIST:  printf(" if event_wait_list is NULL and num_events_in_wait_list > 0, or event_wait_list is not NULL and num_events_in_wait_list is 0, or if event objects in event_wait_list are not valid events..\n");
			break;
		case CL_OUT_OF_HOST_MEMORY:  printf(" if there is a failure to allocate resources required by the OpenCL implementation on the host..\n");
			break;
		}
		printf("Error to create context");
		exit(1);
	}
	clWaitForEvents(1, &event);
	clFinish(queue);
	return event;
}

cl_context createContext(cl_device_id deviceID)
{
	cl_int errcode_ret;
	cl_context context = clCreateContext(NULL, 1, &deviceID, NULL, NULL, &errcode_ret);
	if (errcode_ret != CL_SUCCESS)
	{
		printf("Error to create context");
		exit(1);
	}
	return context;
}

cl_kernel createKernel(cl_program program, const char* programName)
{
	cl_int errcode_ret;
	errcode_ret = CL_SUCCESS;
	cl_kernel kernel = clCreateKernel(program, programName, &errcode_ret);
	if (errcode_ret != CL_SUCCESS)
	{
		printf("Error to create Kernel");
		return 0;
	}
}
