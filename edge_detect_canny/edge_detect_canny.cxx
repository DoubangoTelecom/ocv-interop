#include "../compv_opencv.h"

int _tmain(int argc, _TCHAR* argv[])
{
	COMPV_CHECK_CODE_ASSERT(itp_init());

	char escapeKey = '\0';

	std::vector< KeyPoint > keypoints;
	Mat frame;
	Mat src_gray;
	Mat grad;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	cv::VideoCapture cap(IMPL_CAMERA_ID);
	cap.set(CV_CAP_PROP_FPS, IMPL_CAMERA_FRAME_RATE);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, IMPL_CAMERA_FRAME_WIDTH);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, IMPL_CAMERA_FRAME_HEIGHT);
	COMPV_ASSERT(cap.isOpened());

	do {
		cap >> frame;
		//frame = imread("C:/Projects/GitHub/compv/tests/girl.jpg"); // line_hz.jpg // girl.jpg // circle.jpg // lena

		//GaussianBlur(frame, frame, Size(3, 3), 0, 0, BORDER_DEFAULT);
		COMPV_CHECK_CODE_ASSERT(itp_imageBgrToGrayscale(frame, src_gray));
		
		Mat grad_x, grad_y;
		Mat abs_grad_x, abs_grad_y, grad;

		uint64_t time0 = CompVTime::getNowMills();

#if 0
		itp_canny(src_gray, grad);
#elif 1
		Mat grad_otsu;
		double otsu_thresh_val = cv::threshold(
			src_gray, grad_otsu, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU
			);
		imshow("Otsu", grad_otsu);
		double high_thresh_val = otsu_thresh_val,
			lower_thresh_val = otsu_thresh_val * 0.5;
		cv::Canny(src_gray, grad, lower_thresh_val, high_thresh_val);
		//Canny(src_gray, grad, 100, 50, 3); // 170 90
#else
		cv::Scalar mean = cv::mean(src_gray);
		//double low = 0.52*mean.val[0], high = 1.048*mean.val[0];
		double low = 0.66*mean.val[0], high = 1.33*mean.val[0];
		cv::Canny(src_gray, grad, low, high);
#endif

		//Canny(grayscale, imageEdges, 200, 100, 3);
		//Test(grayscale, imageEdges);
		//Scharr(grayscale, imageEdges, CV_16S, 1, 0); // Gradient X
		//Scharr(grayscale, imageEdges, CV_16S, 0, 1); // Gradient Y
		uint64_t time1 = CompVTime::getNowMills();
		COMPV_DEBUG_INFO("Canny time: %llu", (time1 - time0));
		imshow("Input", frame);
		imshow("Output", grad);
	} while ((escapeKey = cvWaitKey(1000 / IMPL_CAMERA_FRAME_RATE)) != 'q');

	return 0;
}

