#include "../compv_opencv.h"

#define JPEG_LEFT	"panorama_left.jpg"
#define JPEG_RIGHT "panorama_right.jpg"

int _tmain(int argc, _TCHAR* argv[])
{
	char escapeKey = '\0';

	COMPV_CHECK_CODE_ASSERT(itp_init());

	// Read images
	Mat imageLeft = imread(JPEG_LEFT);
	Mat imageRight = imread(JPEG_RIGHT);
	COMPV_ASSERT(imageLeft.data && imageRight.data);
	imshow("left image", imageLeft);
	imshow("right image", imageRight);

	// Convert images to grayscale
	Mat imageLeftGray;
	Mat imageRightGray;
	cvtColor(imageLeft, imageLeftGray, CV_RGB2GRAY);
	cvtColor(imageRight, imageRightGray, CV_RGB2GRAY);
	COMPV_ASSERT(imageLeftGray.data && imageRightGray.data);
	
	// Detect and describe keypoints
	//!\  Detector must be attached to descriptor only if describe() use the same input as the previous detect()
	std::vector< KeyPoint > keypoints_object, keypoints_scene;
	Mat descriptors_object, descriptors_scene;
	IMPL_DETECTOR_PTR detector = NULL;
	IMPL_DESCRIPTOR_PTR descriptor = NULL;
	COMPV_CHECK_CODE_ASSERT(itp_createDetector(detector));
	COMPV_CHECK_CODE_ASSERT(itp_createDescriptor(descriptor, detector));
	COMPV_CHECK_CODE_ASSERT(itp_detect(imageLeftGray, detector, keypoints_object));
	COMPV_CHECK_CODE_ASSERT(itp_describe(imageLeftGray, descriptor, keypoints_object, descriptors_object));
	COMPV_CHECK_CODE_ASSERT(itp_detect(imageRightGray, detector, keypoints_scene));
	COMPV_CHECK_CODE_ASSERT(itp_describe(imageRightGray, descriptor, keypoints_scene, descriptors_scene));

	// Match descriptors
	std::vector< DMatch > good_matches;
	IMPL_MATCHER_PTR matcher;
	COMPV_CHECK_CODE_ASSERT(itp_createMatcher(matcher));
	COMPV_CHECK_CODE_ASSERT(itp_match(matcher, descriptors_scene, descriptors_object, good_matches));
	std::vector< Point2f > obj;
	std::vector< Point2f > scene;
	for (int i = 0; i < good_matches.size(); i++) {
		obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
	}

	// Compute homography
	Mat H;
	COMPV_CHECK_CODE_ASSERT(itp_homography(obj, scene, H));

	// Warp
	cv::Mat result;
	warpPerspective(imageLeft, result, H, cv::Size(imageLeft.cols + imageRight.cols, imageLeft.rows));

	// Blend
	cv::Mat half(result, cv::Rect(0, 0, imageRight.cols, imageRight.rows));
	imageRight.copyTo(half);
	imshow("Result", result);
	
	waitKey(0);
	return 0;
}

