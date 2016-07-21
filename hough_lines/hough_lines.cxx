#include "../compv_opencv.h"

#define CANNY_LOW			1.8f
#define CANNY_HIGH			CANNY_LOW*2.f
#define CANNY_KERNEL_SIZE	3
#define HOUGH_RHO			1.f
#define HOUGH_THETA			kfMathTrigPiOver180 // 1rad
#define HOUGH_THRESHOLD		100

int _tmain(int argc, _TCHAR* argv[])
{
	COMPV_CHECK_CODE_ASSERT(itp_init());

	IMPL_CANNY_PTR canny;
	IMPL_HOUGHSTD_PTR houghStd;
	Mat src;
	Mat src_gray;
	Mat grad;
	Mat hough;
	uint64_t time0, time1;

	COMPV_CHECK_CODE_ASSERT(itp_createCanny(canny, CANNY_LOW, CANNY_HIGH, CANNY_KERNEL_SIZE));
	COMPV_CHECK_CODE_ASSERT(itp_createHoughStd(houghStd, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD));

	cv::VideoCapture cap(IMPL_CAMERA_ID);
	cap.set(CV_CAP_PROP_FPS, IMPL_CAMERA_FRAME_RATE);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, IMPL_CAMERA_FRAME_WIDTH);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, IMPL_CAMERA_FRAME_HEIGHT);
	COMPV_ASSERT(cap.isOpened());

	do {
		//cap >> src;
		// "C:/Projects/GitHub/compv/tests/equirectangular.jpg" // line_hz.jpg // girl.jpg // circle.jpg // lena // building // equirectangular
		src = imread("C:/Projects/GitHub/compv/tests/building.jpg"); // line_hz.jpg // girl.jpg // circle.jpg // lena // building // equirectangular

#if 0 // Canny is slow when next code is used. Strange because there is no relation.
		COMPV_CHECK_CODE_ASSERT(itp_imageBgrToGrayscale(frame, src_gray));
#else
		cvtColor(src, src_gray, CV_BGR2GRAY);
#endif

		COMPV_CHECK_CODE_ASSERT(itp_canny(canny, src_gray, grad));

#if 0
		vector<Vec4i> lines;
		time0 = CompVTime::getNowMills();
		HoughLinesP(grad, lines, 1, CV_PI / 180, 80, 30, 10);
		time1 = CompVTime::getNowMills();
		COMPV_DEBUG_INFO("HoughLinesP time: %llu", (time1 - time0));
		cvtColor(grad, hough, CV_GRAY2BGR);
		for (size_t i = 0; i < lines.size(); i++) {
			line(hough, Point(lines[i][0], lines[i][1]),
				Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 1, CV_AA);
		}
#else
		vector<Vec2f> lines;
		COMPV_CHECK_CODE_ASSERT(itp_houghStdLines(houghStd, grad, lines));
		cvtColor(grad, hough, CV_GRAY2BGR);
		for (size_t i = 0; i < lines.size(); i++) {
			float rho = lines[i][0];
			float theta = lines[i][1];
			double a = cos(theta), b = sin(theta);
			double x0 = a*rho, y0 = b*rho;
			Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
			Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));
			line(hough, pt1, pt2, Scalar(0, 0, 255), 1, CV_AA);
		}
#endif
		imshow("Input", src);
		imshow("canny", grad);
		imshow("Lines", hough);
	} while (cvWaitKey(1000 / IMPL_CAMERA_FRAME_RATE) != 'q');

	return 0;
}
