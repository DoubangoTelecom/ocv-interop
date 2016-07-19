#include "../compv_opencv.h"

#define ADAS_CANNY_LOW			.2f
#define ADAS_CANNY_HIGH			ADAS_CANNY_LOW*1.5f
#define ADAS_CANNY_KERNEL_SIZE	3
#define ADAS_VIDEO_URL			"C:/Projects/line-detection/line-detect-india/finalll111.wmv"
#define ADAS_VIDEO_SNAP_URL		"C:/Users/dmi/Pictures/vlcsnap-2016-07-13-22h51m40s373.png"

// C:/Projects/line-detection/line-detect-india/finalll111.wmv
// C:/Projects/line-detection/highway45.mp4
// C:/Projects/line-detection/raw-washington1.mp4

int _tmain(int argc, _TCHAR* argv[])
{
	COMPV_CHECK_CODE_ASSERT(itp_init());

	IMPL_CANNY_PTR canny;
	Mat frame;
	Mat src_gray;
	Mat grad;

	COMPV_CHECK_CODE_ASSERT(itp_createCanny(canny, ADAS_CANNY_LOW, ADAS_CANNY_HIGH, ADAS_CANNY_KERNEL_SIZE));

	VideoCapture cap(ADAS_VIDEO_URL);

	do {
#if 1
		cap.read(frame);
		if (!frame.size().width) {
			COMPV_DEBUG_INFO("End of the video");
			break;
		}
#else
		frame = imread(ADAS_VIDEO_SNAP_URL);
#endif
		GaussianBlur(frame, frame, Size(17, 17), 0, 0, BORDER_DEFAULT);

#if 0 // Canny is slow when next code is used. Strange because there is no relation.
		COMPV_CHECK_CODE_ASSERT(itp_imageBgrToGrayscale(frame, src_gray));
#else
		cvtColor(frame, src_gray, CV_BGR2GRAY);
#endif

		double otsu_thresh_val = cv::threshold(
			src_gray, grad, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU
			);
		double high_thresh_val = otsu_thresh_val;
		double lower_thresh_val = otsu_thresh_val * 0.5;
		//canny->setOtsu((int)lower_thresh_val, (int)high_thresh_val);

		//uint64_t time0 = CompVTime::getNowMills();
		COMPV_CHECK_CODE_ASSERT(itp_canny(canny, src_gray, grad));
		//uint64_t time1 = CompVTime::getNowMills();
		//COMPV_DEBUG_INFO("Canny time: %llu", (time1 - time0));
		imshow("Input", frame);
		imshow("Output", grad);
	} while (cvWaitKey(1000 / IMPL_CAMERA_FRAME_RATE) != 'q');

	cap.release();
	cvWaitKey(0);

	return 0;
}
