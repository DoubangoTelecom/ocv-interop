#include "../compv_opencv.h"

#define CANNY_LOW			0.8f
#define CANNY_HIGH			CANNY_LOW*2.f
#define CANNY_KERNEL_SIZE	3

int _tmain(int argc, _TCHAR* argv[])
{
	COMPV_CHECK_CODE_ASSERT(itp_init());

	IMPL_CANNY_PTR canny;
	Mat frame;
	Mat src_gray;
	Mat grad;

	COMPV_CHECK_CODE_ASSERT(itp_createCanny(canny, CANNY_LOW, CANNY_HIGH, CANNY_KERNEL_SIZE));

	cv::VideoCapture cap(IMPL_CAMERA_ID);
	cap.set(CV_CAP_PROP_FPS, IMPL_CAMERA_FRAME_RATE);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, IMPL_CAMERA_FRAME_WIDTH);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, IMPL_CAMERA_FRAME_HEIGHT);
	COMPV_ASSERT(cap.isOpened());

	do {
		cap >> frame;
		//frame = imread("C:/Projects/GitHub/compv/tests/girl.jpg"); // line_hz.jpg // girl.jpg // circle.jpg // lena
		GaussianBlur(frame, frame, Size(3, 3), 0, 0, BORDER_DEFAULT);

#if 0 // Canny is slow when next code is used. Strange because there is no relation.
		COMPV_CHECK_CODE_ASSERT(itp_imageBgrToGrayscale(frame, src_gray));
#else
		cvtColor(frame, src_gray, CV_BGR2GRAY);
#endif
		
		//uint64_t time0 = CompVTime::getNowMills();
		COMPV_CHECK_CODE_ASSERT(itp_canny(canny, src_gray, grad));
		//uint64_t time1 = CompVTime::getNowMills();
		//COMPV_DEBUG_INFO("Canny time: %llu", (time1 - time0));
		imshow("Input", frame);
		imshow("Output", grad);
	} while (cvWaitKey(1000 / IMPL_CAMERA_FRAME_RATE) != 'q');

	return 0;
}

