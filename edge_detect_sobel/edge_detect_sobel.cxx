#include "../compv_opencv.h"

int _tmain(int argc, _TCHAR* argv[])
{
	COMPV_CHECK_CODE_ASSERT(itp_init());

	char escapeKey = '\0';

	cv::VideoCapture cap(IMPL_CAMERA_ID);
	cap.set(CV_CAP_PROP_FPS, IMPL_CAMERA_FRAME_RATE);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, IMPL_CAMERA_FRAME_WIDTH);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, IMPL_CAMERA_FRAME_HEIGHT);
	COMPV_ASSERT(cap.isOpened());

	Mat frame;
	Mat grad;

	do {
		cap >> frame;
		COMPV_CHECK_CODE_ASSERT(itp_imageBgrToGrayscale(frame, frame));
		COMPV_CHECK_CODE_ASSERT(itp_sobel(frame, grad));
		imshow("Input", frame);
		imshow("Output", grad);
	} while ((escapeKey = cvWaitKey(1000 / IMPL_CAMERA_FRAME_RATE)) != 'q');

	return 0;
}

