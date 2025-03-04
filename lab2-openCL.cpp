#include "myOpencl.h"

template <typename T>
Mat createMat(T* data, int rows, int cols, int chs = 1) {
	// Create Mat from buffer 
	Mat mat(rows, cols, CV_MAKETYPE(DataType<T>::type, chs));
	memcpy(mat.data, data, rows * cols * chs * sizeof(T));
	return mat;
}

int main()
{
	const char* filename = "forest.bmp";

	cl_device_id deviceID = getDeviceInfo();

	// 5. Создание контекста
	cl_context context = createContext(deviceID);


	// 6. Создание очереди команд
	cl_command_queue queue = createQueue(deviceID, context, 2);

	// 7. Создание программы
	cl_int errcode_ret = CL_SUCCESS;
	 cl_program program = build_program(context, deviceID, IMG_TEST_PROGRAM);

	// 9. Получение ядра
	cl_kernel kernel = createKernel(program, "imgTest");

	// 10. Создание избражения
	errcode_ret = 0;

	cl_mem image1, image2;
	Mat image = createImages(filename, &image1, &image2, context);

	// 11. Установка изображения в качестве аргумента ядра
	errcode_ret = clSetKernelArg(kernel, 0, sizeof(image1), (void*)&image1);
	errcode_ret |= clSetKernelArg(kernel, 1, sizeof(image2), (void*)&image2);
	if (errcode_ret != CL_SUCCESS) {
		printf("Error to set kernel arg");
		return 0;
	}

	// 12. Запуск ядра
	size_t global_work_size[2] = {image.cols, image.rows};
	executeKernel(global_work_size, NULL, queue, kernel, 2);

	// 13. Копирование изображения  в память управляющего узла
	cl_event event[5];

	size_t origin[] = { 0,0,0 }; // Defines the offset in pixels in the image from where to write.
	size_t region[] = { image.cols, image.rows, 1 }; // Size of object to be transferred
	const int size1 = image.cols * image.rows * 4;
	unsigned char* buffer_result = (unsigned char*)calloc(size1, sizeof(unsigned char));

	errcode_ret = 0;
	// read image in buffer
	errcode_ret = clEnqueueReadImage(queue, image2, CL_TRUE, origin, region, 0, 0, buffer_result, 0, NULL, &event[0]);
	if (errcode_ret != CL_SUCCESS)
	{
		printf("Error to create context");
		return 0;
	}
	// write image to cv::mat
	Mat image_result = createMat<unsigned char>(buffer_result, image.rows, image.cols, 4);

	// 14. Использование результатов
	namedWindow("Display window result", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Display window result", image_result);        // Show our image inside it.
	waitKey(0);

	// 15. Удаление объектов и освобождение памяти управляющего узла
	if (buffer_result) free(buffer_result);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	return  0;
}



//#include <opencv2/opencv.hpp>
//#include <iostream>
//#include <io.h>
//
//
//int main() {
//    // Путь к изображению
//    std::string imagePath = "D:\\rita\\studyspace\\lab1-openCL\\forest.bmp";
//
//    if (_access(imagePath.c_str(), 0) != 0) { // 0 проверяет существование файла
//        std::cout << "File does not exist: " << imagePath << std::endl;
//        return -1;
//    }
//
//    // Загружаем изображение
//    cv::Mat image = cv::imread(imagePath);
//    if (image.empty()) {
//        std::cout << "Failed to load image from path: " << imagePath << std::endl;
//        return -1;
//    }
//
//    // Отображаем изображение
//    cv::imshow("Test", image);
//
//    // Ждем нажатия клавиши
//    cv::waitKey(0);
//
//    return 0;
//}