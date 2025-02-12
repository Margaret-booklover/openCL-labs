#define _CRT_SECURE_NO_WARNINGS 1
#pragma once
#include <CL/opencl.h>
#include <CL/cl_platform.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define PROGRAM_FILE "kernel.cl"
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <iostream>
#include "myOpencl.h"
using namespace std;

cl_command_queue createQueue(cl_device_id deviceID, cl_context context);
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename);
cl_device_id getDeviceInfo();
cl_context createContext(cl_device_id deviceID);
cl_event executeKernel(size_t uGlobalWorkSize, cl_command_queue queue, cl_kernel kernel);
cl_kernel createKernel(cl_program program, const char* programName);
