#include "../compv_opencv.h"

int _tmain(int argc, _TCHAR* argv[])
{
	char escapeKey = '\0';

	COMPV_CHECK_CODE_ASSERT(itp_init());

	std::vector< KeyPoint > keypoints;
	Mat frame;
	Mat imageGray;
	IMPL_DETECTOR_PTR detector = NULL;

	COMPV_CHECK_CODE_ASSERT(itp_createDetector(detector));

	cv::VideoCapture cap(IMPL_CAMERA_ID);
	cap.set(CV_CAP_PROP_FPS, IMPL_CAMERA_FRAME_RATE);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, IMPL_CAMERA_FRAME_WIDTH);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, IMPL_CAMERA_FRAME_HEIGHT);
	COMPV_ASSERT(cap.isOpened());

	do {
		cap >> frame;
		cvtColor(frame, imageGray, CV_RGB2GRAY);

		COMPV_CHECK_CODE_ASSERT(itp_detect(imageGray, detector, keypoints));
		drawKeypoints(imageGray, keypoints, imageGray);

		imshow("Input", frame);
		imshow("Output", imageGray);

	} while ((escapeKey = cvWaitKey(1000 / IMPL_CAMERA_FRAME_RATE)) != 'q');

	return 0;
}

