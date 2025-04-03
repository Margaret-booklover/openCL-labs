#define _CRT_SECURE_NO_WARNINGS 1
#pragma once
#include <CL/opencl.h>
#include <CL/cl_platform.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define PROGRAM_FILE "kernel.cl"
#define PROGRAM_FILE2 "matMul.cl"
#define IMG_TEST_PROGRAM "imgTest.cl"
#define CONVOLUTION_PROGRAM "convolution.cl"
#define CONVOLUTION_PROGRAM1 "localCon.cl"
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <cmath>
using namespace cv;
using namespace std;

cl_command_queue createQueue(cl_device_id deviceID, cl_context context, int propId);
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename);
cl_device_id getDeviceInfo();
cl_context createContext(cl_device_id deviceID);
cl_event executeKernel(size_t* uGlobalWorkSize, size_t* uLocalWorkSize, cl_command_queue queue, cl_kernel kernel, cl_uint work_dim);
cl_kernel createKernel(cl_program program, const char* programName);
Mat createImages(const char* filename, cl_mem* image1, cl_mem* image2, cl_context context);
Mat createRGBAImages(const char* filename, cl_mem* image1, cl_mem* image2, cl_context context);
