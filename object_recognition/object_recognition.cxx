#include "../compv_opencv.h"

#define JPEG_IMG  "opengl_programming_guide_8th_edition.jpg"

int _tmain(int argc, _TCHAR* argv[])
{
	COMPV_CHECK_CODE_ASSERT(itp_init());

	char escapeKey = '\0';

	Mat trainImage = imread(JPEG_IMG, CV_LOAD_IMAGE_GRAYSCALE);
	COMPV_ASSERT(trainImage.data != NULL);

	namedWindow("Good Matches", CV_WINDOW_AUTOSIZE);

	// Detect and describe keypoints
	//!\  Detector must be attached to descriptor only if describe() use the same input as the previous detect()
	vector<KeyPoint> trainKeypoints;
	Mat trainDescriptors;
	IMPL_DETECTOR_PTR detector = NULL;
	IMPL_DESCRIPTOR_PTR descriptor = NULL;
	COMPV_CHECK_CODE_ASSERT(itp_createDetector(detector));
	COMPV_CHECK_CODE_ASSERT(itp_createDescriptor(descriptor, detector));
	COMPV_CHECK_CODE_ASSERT(itp_detect(trainImage, detector, trainKeypoints));
	COMPV_CHECK_CODE_ASSERT(itp_describe(trainImage, descriptor, trainKeypoints, trainDescriptors));

	// Create matcher
	IMPL_MATCHER_PTR matcher;
	COMPV_CHECK_CODE_ASSERT(itp_createMatcher(matcher));
	

	//Object corner points for plotting box
	vector<Point2f> trainCorners(4);
	trainCorners[0] = cvPoint(0, 0);
	trainCorners[1] = cvPoint(trainImage.cols, 0);
	trainCorners[2] = cvPoint(trainImage.cols, trainImage.rows);
	trainCorners[3] = cvPoint(0, trainImage.rows);
	
	cv::VideoCapture cap(IMPL_CAMERA_ID);
	cap.set(CV_CAP_PROP_FPS, IMPL_CAMERA_FRAME_RATE);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, IMPL_CAMERA_FRAME_WIDTH);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, IMPL_CAMERA_FRAME_HEIGHT);
	COMPV_ASSERT(cap.isOpened());

	Mat queryDescriptors, img_matches, H;
	vector<KeyPoint> queryKeypoints;
	vector<vector<DMatch > > matches;
	vector<DMatch > good_matches;
	vector<Point2f> scene_corners(4);
	vector<Point2f> obj;
	vector<Point2f> scene;

	do {
		Mat queryFrame;
		Mat queryImage;
		cap >> queryFrame;
		if (queryFrame.dims < 2) {
			COMPV_DEBUG_ERROR("Failed to capture queryFrame from camera");
			continue;
		}

		// Convert queryImage to grayscale
		COMPV_CHECK_CODE_ASSERT(itp_imageBgrToGrayscale(queryFrame, queryImage));
		
		// Detect and describe keypoints
		COMPV_CHECK_CODE_ASSERT(itp_detect(queryImage, detector, queryKeypoints));
		COMPV_CHECK_CODE_ASSERT(itp_describe(queryImage, descriptor, queryKeypoints, queryDescriptors));

		// Match descriptions
		COMPV_CHECK_CODE_ASSERT(itp_match(matcher, queryDescriptors, trainDescriptors, good_matches));
		
		// Draw matches
		//uint64_t timeStart = CompVTime::getNowMills();
		drawMatches(trainImage, trainKeypoints, /*queryImage*/queryFrame, queryKeypoints, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		//uint64_t timeEnd = CompVTime::getNowMills();
		//COMPV_DEBUG_INFO("drawMatches=%llu", (timeEnd - timeStart));

		if (good_matches.size() >= thresholdGoodMatches) {
			putText(img_matches, "Object Recognized!", cvPoint(10, 50), FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(0, 0, 250), 1, CV_AA);
			obj.clear();
			scene.clear();
			for (unsigned int i = 0; i < good_matches.size(); i++) {
				obj.push_back(trainKeypoints[good_matches[i].queryIdx].pt);
				scene.push_back(queryKeypoints[good_matches[i].trainIdx].pt);
			}

			// Find homography
			COMPV_CHECK_CODE_ASSERT(itp_homography(obj, scene, H));

			// Perspecive transform using homography matrix
			COMPV_CHECK_CODE_ASSERT(itp_perspectiveTransform(trainCorners, scene_corners, H));

			// Draw lines between the corners (the mapped trainImage in the scene queryImage )
			line(img_matches, scene_corners[0] + Point2f((float)trainImage.cols, 0.f), scene_corners[1] + Point2f((float)trainImage.cols, 0), Scalar(0, 255, 0), 4);
			line(img_matches, scene_corners[1] + Point2f((float)trainImage.cols, 0.f), scene_corners[2] + Point2f((float)trainImage.cols, 0), Scalar(0, 255, 0), 4);
			line(img_matches, scene_corners[2] + Point2f((float)trainImage.cols, 0.f), scene_corners[3] + Point2f((float)trainImage.cols, 0), Scalar(0, 255, 0), 4);
			line(img_matches, scene_corners[3] + Point2f((float)trainImage.cols, 0.f), scene_corners[0] + Point2f((float)trainImage.cols, 0), Scalar(0, 255, 0), 4);
		}

		imshow("Good Matches", img_matches);

	} while ((escapeKey = cvWaitKey(1000 / IMPL_CAMERA_FRAME_RATE)) != 'q');

	cap.release();

	return 0;
}

