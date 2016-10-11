#include <compv/compv_api.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include <stdio.h>
#include <iostream>
#if COMPV_OS_WINDOWS
#	include <tchar.h>
#endif

using namespace cv;
using namespace compv;
using namespace std;

struct cannyThresholds {
	float low;
	float high;
	int32_t kernelSize;
};

struct houghStd {
	float rho;
	float theta;
	int32_t threshold;
};

/* ===================================================================== */
#define DECL_DETECTOR_FAST			0
#define DECL_DETECTOR_STAR			1
#define DECL_DETECTOR_SIFT			2
#define DECL_DETECTOR_SURF			3
#define DECL_DETECTOR_ORB			4
#define DECL_DETECTOR_BRISK			5
#define DECL_DETECTOR_MSER			6
#define DECL_DETECTOR_ORB_COMPV		7

#define DECL_EXTRACTOR_SIFT			0
#define DECL_EXTRACTOR_SURF			1
#define DECL_EXTRACTOR_BRIEF		2
#define DECL_EXTRACTOR_BRISK		3
#define DECL_EXTRACTOR_ORB			4
#define DECL_EXTRACTOR_FREAK		5
#define DECL_EXTRACTOR_ORB_COMPV	6

#define DECL_HOMOGRAPHY_COMPV		0
#define DECL_HOMOGRAPHY_OPENCV		1

#define DECL_PERSPTRANSFORM_COMPV	0 // perspectiveTransform()
#define DECL_PERSPTRANSFORM_OPENCV	1 // perspectiveTransform()

#define DECL_MATCHER_BRUTE_FORCE				0
#define DECL_MATCHER_BRUTE_FORCE_HAMMING		1 // ORB WTA_K = 2, outputs on 1bit=0/1
#define DECL_MATCHER_BRUTE_FORCE_HAMMING2		2 // ORB WTA_K = 3 or 4, outputs on 2bits=0/1/2/3
#define DECL_MATCHER_FLANN						3
#define DECL_MATCHER_BRUTE_FORCE_COMPV			4

#define DECL_CANNY_COMPV		0
#define DECL_CANNY_OPENCV		1

#define DECL_HOUGHSTD_COMPV		0
#define DECL_HOUGHSTD_OPENCV	1

#define DECL_IMGCONV_COMPV		0
#define DECL_IMGCONV_OPENCV		1

#define DECL_PRESET_NONE		0 // Ignore preset
#define DECL_PRESET_ORB			1
#define DECL_PRESET_SURF		2
#define DECL_PRESET_SIFT		3
#define DECL_PRESET_COMPV		4

#define DECL_TYPE_DOUBLE		0
#define DECL_TYPE_FLOAT			1


/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

#define IMPL_TYPE			DECL_TYPE_DOUBLE
#define IMPL_PRESET			DECL_PRESET_COMPV

#define IMPL_CAMERA_ID					0
#define IMPL_CAMERA_FRAME_RATE			120
#define IMPL_CAMERA_FRAME_WIDTH			1280 // 1920
#define IMPL_CAMERA_FRAME_HEIGHT		720 // 1080

#define IMPL_PYRAMID_LEVELS				8
#define IMPL_MAX_FEATURES				2000 // 2000 // default is 500 but this dosen't provide good result (OpenCV default value: 500) - 500 is OK for image stitching
#define IMPL_CROSS_CHECK				true // must be true only when (KNN == 1)
#define IMPL_KNN						2
#define IMPL_NMS						true
#define IMPL_FAST_THRESHOLD				20
#define IMPL_MIN_HESS					400

#define IMPL_NUM_THREADS			COMPV_NUM_THREADS_BEST
#define IMPL_ENABLE_INTRINSICS		true
#define IMPL_ENABLE_ASM				true
#define IMPL_ENABLE_FIXED_POINT		true
#define IMPL_ENABLE_TESTING_MODE	false
#define IMPL_CPU_DISABLE_FLAGS		kCpuFlagNone

/* ------------------------------------------------------------------------ */

#if IMPL_TYPE == DECL_TYPE_DOUBLE
#	define TYPE_COMPV				double
#	define TYPE_OPENCV				CV_64F
#elif IMPL_TYPE == DECL_TYPE_FLOAT
#	define TYPE_COMPV				float
#	define TYPE_OPENCV				CV_32F
#else
#	error "Not implemented"
#endif

#if IMPL_PRESET	== DECL_PRESET_ORB
#	define IMPL_DETECTOR		DECL_DETECTOR_ORB
#	define IMPL_EXTRACTOR		DECL_EXTRACTOR_ORB
#	define IMPL_MATCHER			DECL_MATCHER_BRUTE_FORCE_HAMMING
#	define IMPL_HOMOGRAPHY		DECL_HOMOGRAPHY_OPENCV
#	define IMPL_PERSPTRANSFORM	DECL_PERSPTRANSFORM_OPENCV
#	define IMPL_CANNY			DECL_CANNY_OPENCV
#	define IMPL_HOUGHSTD		DECL_HOUGHSTD_OPENCV
#	define IMPL_IMGCONV			DECL_IMGCONV_OPENCV
#elif IMPL_PRESET == DECL_PRESET_COMPV
#	define IMPL_DETECTOR		DECL_DETECTOR_ORB_COMPV
#	define IMPL_EXTRACTOR		DECL_EXTRACTOR_ORB_COMPV
#	define IMPL_MATCHER			DECL_MATCHER_BRUTE_FORCE_COMPV
#	define IMPL_HOMOGRAPHY		DECL_HOMOGRAPHY_COMPV
#	define IMPL_PERSPTRANSFORM	DECL_PERSPTRANSFORM_COMPV
#	define IMPL_CANNY			DECL_CANNY_COMPV
#	define IMPL_HOUGHSTD		DECL_HOUGHSTD_COMPV
#	define IMPL_IMGCONV			DECL_IMGCONV_COMPV //!\ CompV image wrapping and copying is very slow...but compv impl. outputs better quality than opencv
#elif IMPL_PRESET == DECL_PRESET_SURF
#	define IMPL_DETECTOR		DECL_DETECTOR_SURF
#	define IMPL_EXTRACTOR		DECL_EXTRACTOR_SURF
#	define IMPL_MATCHER			DECL_MATCHER_BRUTE_FORCE
#	define IMPL_HOMOGRAPHY		DECL_HOMOGRAPHY_OPENCV
#	define IMPL_PERSPTRANSFORM	DECL_PERSPTRANSFORM_OPENCV
#	define IMPL_CANNY			DECL_CANNY_OPENCV
#	define IMPL_HOUGHSTD		DECL_HOUGHSTD_OPENCV
#	define IMPL_IMGCONV			DECL_IMGCONV_OPENCV
#elif IMPL_PRESET == DECL_PRESET_SIFT
#	define IMPL_DETECTOR		DECL_DETECTOR_SIFT
#	define IMPL_EXTRACTOR		DECL_EXTRACTOR_SIFT
#	define IMPL_MATCHER			DECL_MATCHER_BRUTE_FORCE
#	define IMPL_HOMOGRAPHY		DECL_HOMOGRAPHY_OPENCV
#	define IMPL_PERSPTRANSFORM	DECL_PERSPTRANSFORM_OPENCV
#	define IMPL_CANNY			DECL_CANNY_OPENCV
#	define IMPL_HOUGHSTD		DECL_HOUGHSTD_OPENCV
#	define IMPL_IMGCONV			DECL_IMGCONV_OPENCV
#else // NONE
#	define IMPL_DETECTOR		DECL_DETECTOR_ORB_COMPV
#	define IMPL_EXTRACTOR		DECL_EXTRACTOR_ORB_COMPV
#	define IMPL_MATCHER			DECL_MATCHER_BRUTE_FORCE_COMPV
#	define IMPL_HOMOGRAPHY		DECL_HOMOGRAPHY_COMPV
#	define IMPL_PERSPTRANSFORM	DECL_PERSPTRANSFORM_COMPV
#	define IMPL_CANNY			DECL_CANNY_OPENCV
#	define IMPL_HOUGHSTD		DECL_HOUGHSTD_OPENCV
#	define IMPL_IMGCONV			DECL_IMGCONV_OPENCV
#endif



#if IMPL_DETECTOR == DECL_DETECTOR_ORB_COMPV
#	define IMPL_DETECTOR_PTR	CompVPtr<CompVCornerDete* >
#else
#	define IMPL_DETECTOR_PTR	cv::Ptr<cv::FeatureDetector>
#endif
#if IMPL_EXTRACTOR == DECL_EXTRACTOR_ORB_COMPV
#	define IMPL_DESCRIPTOR_PTR	CompVPtr<CompVCornerDesc* >
#else
#	define IMPL_DESCRIPTOR_PTR	cv::Ptr<cv::DescriptorExtractor>
#endif
#if IMPL_MATCHER == DECL_MATCHER_BRUTE_FORCE_COMPV
#	define IMPL_MATCHER_PTR		CompVPtr<CompVMatcher* >
#else
#	define IMPL_MATCHER_PTR		cv::Ptr<cv::DescriptorMatcher>
#endif
#if IMPL_CANNY == DECL_CANNY_COMPV
#	define IMPL_CANNY_PTR		CompVPtr<CompVEdgeDete* >
#else
#	define IMPL_CANNY_PTR		cannyThresholds
#endif
#if IMPL_HOUGHSTD == DECL_HOUGHSTD_COMPV
#	define IMPL_HOUGHSTD_PTR	CompVPtr<CompVHough* >
#else
#	define IMPL_HOUGHSTD_PTR	houghStd
#endif

#if IMPL_EXTRACTOR == DECL_EXTRACTOR_SURF || IMPL_EXTRACTOR == DECL_EXTRACTOR_SIFT
#	define ratioTestKNN			0.6 // http://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf#page=20
#else
#	define ratioTestKNN			0.67
#endif
#define thresholdGoodMatches	8

/* itp_init() */
static COMPV_ERROR_CODE itp_init()
{
	setUseOptimized(true); 

	CompVDebugMgr::setLevel(COMPV_DEBUG_LEVEL_INFO);
	COMPV_CHECK_CODE_RETURN(CompVEngine::init(IMPL_NUM_THREADS));
	COMPV_CHECK_CODE_RETURN(CompVEngine::setTestingModeEnabled(IMPL_ENABLE_TESTING_MODE));
	COMPV_CHECK_CODE_RETURN(CompVEngine::setMathFixedPointEnabled(IMPL_ENABLE_FIXED_POINT));
	COMPV_CHECK_CODE_RETURN(CompVCpu::setAsmEnabled(IMPL_ENABLE_ASM));
	COMPV_CHECK_CODE_RETURN(CompVCpu::setIntrinsicsEnabled(IMPL_ENABLE_INTRINSICS));
	COMPV_CHECK_CODE_RETURN(CompVCpu::flagsDisable(IMPL_CPU_DISABLE_FLAGS));

	return COMPV_ERROR_CODE_S_OK;
}

/* itp_createDetector() */
static COMPV_ERROR_CODE itp_imageBgrToGrayscale(const Mat& in, Mat& out)
{
#if IMPL_IMGCONV == DECL_IMGCONV_COMPV // TODO(dmi): Strange: canny slow when using this code (not the case on the MacBookPro)
	// The wrapping and copy is very slow....
	CompVPtr<CompVImage *> img_;
	out = Mat(in.size(), CV_8U);
	COMPV_CHECK_CODE_RETURN(CompVImage::wrap(COMPV_PIXEL_FORMAT_B8G8R8, in.ptr(0), in.size().width, in.size().height, in.size().width, &img_));
	COMPV_CHECK_CODE_RETURN(img_->convert(COMPV_PIXEL_FORMAT_GRAYSCALE, &img_));
	COMPV_CHECK_CODE_RETURN(CompVImage::copy(COMPV_PIXEL_FORMAT_GRAYSCALE, img_->getDataPtr(), img_->getWidth(), img_->getHeight(), img_->getStride(), out.ptr(0), out.size().width, out.size().height, out.size().width));
#else
	cvtColor(in, out, CV_BGR2GRAY);
#endif
	return COMPV_ERROR_CODE_S_OK;
}

/* itp_createDetector() */
static COMPV_ERROR_CODE itp_createDetector(IMPL_DETECTOR_PTR& detector)
{
#if IMPL_DETECTOR == DECL_DETECTOR_ORB_COMPV
	float scaleFactor = 0.83f; // (1 / 1.2)
	int32_t nlevels = IMPL_PYRAMID_LEVELS;
	bool nms = IMPL_NMS;
	int32_t fastThreshold = IMPL_FAST_THRESHOLD;
	int nfeatures = IMPL_MAX_FEATURES;
	COMPV_CHECK_CODE_RETURN(CompVCornerDete::newObj(COMPV_ORB_ID, &detector));
	COMPV_CHECK_CODE_RETURN(detector->set(COMPV_ORB_SET_INT32_FAST_THRESHOLD, &fastThreshold, sizeof(fastThreshold)));
	COMPV_CHECK_CODE_RETURN(detector->set(COMPV_ORB_SET_BOOL_FAST_NON_MAXIMA_SUPP, &nms, sizeof(nms)));
	COMPV_CHECK_CODE_RETURN(detector->set(COMPV_ORB_SET_INT32_PYRAMID_LEVELS, &nlevels, sizeof(nlevels)));
	COMPV_CHECK_CODE_RETURN(detector->set(COMPV_ORB_SET_FLT32_PYRAMID_SCALE_FACTOR, &scaleFactor, sizeof(scaleFactor)));
	COMPV_CHECK_CODE_RETURN(detector->set(COMPV_ORB_SET_INT32_MAX_FEATURES, &nfeatures, sizeof(nfeatures)));
#elif IMPL_DETECTOR == DECL_DETECTOR_ORB
	int nfeatures = IMPL_MAX_FEATURES; // default: 500
	float scaleFactor = 1.2f;
	int nlevels = IMPL_PYRAMID_LEVELS;
	int edgeThreshold = 31;
	int firstLevel = 0;
	int WTA_K = 2; // Number of points to compare (Hamming1 for WTA_K=2, and Hamming2 for WTA_K=3 or 4)
	int scoreType = ORB::FAST_SCORE; // ORB::FAST_SCORE, default: ORB::HARRIS_SCORE;
	int patchSize = 31;
	detector = new cv::OrbFeatureDetector(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize);
#elif IMPL_DETECTOR == DECL_DETECTOR_SURF
	detector = new cv::SurfFeatureDetector(IMPL_MIN_HESS);
#elif IMPL_DETECTOR == DECL_DETECTOR_SIFT
	detector = new cv::SiftFeatureDetector(IMPL_MIN_HESS);
#else
#	error "Not implemented"
#endif

	return COMPV_ERROR_CODE_S_OK;
}

/* itp_detect() */
static COMPV_ERROR_CODE itp_detect(const Mat& grayImage, IMPL_DETECTOR_PTR& detector, std::vector< KeyPoint >& keypoints)
{
	COMPV_CHECK_EXP_RETURN(!detector, COMPV_ERROR_CODE_E_INVALID_PARAMETER);
	keypoints.clear();

#if IMPL_DETECTOR == DECL_DETECTOR_ORB_COMPV
	CompVPtr<CompVBoxInterestPoint* > interestPoints;
	CompVPtr<CompVImage *> queryImage;
	COMPV_CHECK_CODE_ASSERT(CompVImage::wrap(COMPV_PIXEL_FORMAT_GRAYSCALE, grayImage.ptr(0), grayImage.size().width, grayImage.size().height, grayImage.size().width, &queryImage));
	//uint64_t time0 = CompVTime::getNowMills();
	COMPV_CHECK_CODE_ASSERT(detector->process(queryImage, interestPoints));
	//uint64_t time1 = CompVTime::getNowMills();
	//COMPV_DEBUG_INFO("Detect time=%llu", (time1 - time0));
	for (size_t i = 0; i < interestPoints->size(); ++i) {
		const CompVInterestPoint* p = interestPoints->ptr(i);
		keypoints.push_back(KeyPoint(p->x, p->y, p->size, (float)p->orient, (float)p->strength, p->level));
	}
#else
	detector->detect(grayImage, keypoints);
#endif

	return COMPV_ERROR_CODE_S_OK;
}

/* itp_createDescriptor() */
static COMPV_ERROR_CODE itp_createDescriptor(IMPL_DESCRIPTOR_PTR& descriptor, IMPL_DETECTOR_PTR detector = NULL)
{
#if IMPL_EXTRACTOR == DECL_EXTRACTOR_ORB_COMPV
	COMPV_CHECK_CODE_RETURN(CompVCornerDesc::newObj(COMPV_ORB_ID, &descriptor));
#	if IMPL_DETECTOR == DECL_DETECTOR_ORB_COMPV
	COMPV_CHECK_CODE_RETURN(descriptor->attachDete(detector)); // not required (done for performance reasons)
#	endif
#elif IMPL_EXTRACTOR == DECL_EXTRACTOR_ORB
	descriptor = new cv::OrbDescriptorExtractor();
#elif IMPL_EXTRACTOR == DECL_EXTRACTOR_SURF
	descriptor = new cv::SurfDescriptorExtractor();
#elif IMPL_EXTRACTOR == DECL_EXTRACTOR_SIFT
	descriptor = new cv::SiftDescriptorExtractor();
#else
#	error "Not implemented"
#endif
	return COMPV_ERROR_CODE_S_OK;
}

/* itp_describe() */
static COMPV_ERROR_CODE itp_describe(const Mat& grayImage, IMPL_DESCRIPTOR_PTR descriptor, vector<KeyPoint>& keypoints, Mat& imgDescriptor)
{
	COMPV_CHECK_EXP_RETURN(!descriptor, COMPV_ERROR_CODE_E_INVALID_PARAMETER);
	imgDescriptor.resize(0);
#if IMPL_EXTRACTOR == DECL_EXTRACTOR_ORB_COMPV
	if (keypoints.size() > 0) {
		CompVPtr<CompVImage *> queryImage;
		CompVPtr<CompVArray<uint8_t>* > descriptions;
		CompVPtr<CompVBoxInterestPoint* > interestPoints;
		COMPV_CHECK_CODE_RETURN(CompVImage::wrap(COMPV_PIXEL_FORMAT_GRAYSCALE, grayImage.ptr(0), grayImage.size().width, grayImage.size().height, grayImage.size().width, &queryImage));
		COMPV_CHECK_CODE_RETURN(CompVBoxInterestPoint::newObj(&interestPoints));
		CompVInterestPoint ip;
		for (size_t i = 0; i < keypoints.size(); ++i) {
			const KeyPoint* kp = &keypoints[i];
			ip.x = kp->pt.x;
			ip.y = kp->pt.y;
			ip.level = kp->octave;
			ip.orient = kp->angle;
			ip.size = kp->size;
			ip.strength = kp->response;
			interestPoints->push(ip);
		}
		//uint64_t time0 = CompVTime::getNowMills();
		COMPV_CHECK_CODE_RETURN(descriptor->process(queryImage, interestPoints, &descriptions));
		//uint64_t time1 = CompVTime::getNowMills();
		//COMPV_DEBUG_INFO("Describe time=%llu", (time1 - time0));
		COMPV_ASSERT(descriptions->cols() == descriptions->strideInBytes()); // direct copy only if stride == width
		if (descriptions && !descriptions->isEmpty()) {
			imgDescriptor = Mat(Size((int)descriptions->cols(), (int)descriptions->rows()), (int)grayImage.type());
			CompVMem::copy(imgDescriptor.ptr(0), descriptions->ptr(), (descriptions->cols() * descriptions->rows()));
		}
	}
#else
	descriptor->compute(grayImage, keypoints, imgDescriptor);
#endif
	return COMPV_ERROR_CODE_S_OK;
}

/* itp_createMatcher() */
static COMPV_ERROR_CODE itp_createMatcher(IMPL_MATCHER_PTR& matcher)
{
#if IMPL_MATCHER == DECL_MATCHER_BRUTE_FORCE_HAMMING
	matcher = new cv::BFMatcher(cv::NORM_HAMMING, (IMPL_CROSS_CHECK && IMPL_KNN == 1));
#elif IMPL_MATCHER == DECL_MATCHER_BRUTE_FORCE_HAMMING2
	matcher = new cv::BFMatcher(cv::NORM_HAMMING2, (IMPL_CROSS_CHECK && IMPL_KNN == 1));
#elif IMPL_MATCHER == DECL_MATCHER_BRUTE_FORCE
	matcher = new cv::BFMatcher(cv::NORM_L2, (IMPL_CROSS_CHECK && IMPL_KNN == 1));
#elif IMPL_MATCHER == DECL_MATCHER_FLANN
	cv::FlannBasedMatcher matcher;
#elif IMPL_MATCHER == DECL_MATCHER_BRUTE_FORCE_COMPV
	int32_t knn = IMPL_KNN, norm = COMPV_BRUTEFORCE_NORM_HAMMING;
	bool crossCheck = (IMPL_CROSS_CHECK && IMPL_KNN == 1);
	COMPV_CHECK_CODE_RETURN(CompVMatcher::newObj(COMPV_BRUTEFORCE_ID, &matcher));
	COMPV_CHECK_CODE_RETURN(matcher->set(COMPV_BRUTEFORCE_SET_INT32_KNN, &knn, sizeof(knn)));
	COMPV_CHECK_CODE_RETURN(matcher->set(COMPV_BRUTEFORCE_SET_INT32_NORM, &norm, sizeof(norm)));
	COMPV_CHECK_CODE_RETURN(matcher->set(COMPV_BRUTEFORCE_SET_BOOL_CROSS_CHECK, &crossCheck, sizeof(crossCheck)));
#else
#error(not implemented)
#endif
	return COMPV_ERROR_CODE_S_OK;
}

/* itp_match() */
static COMPV_ERROR_CODE itp_match(IMPL_MATCHER_PTR matcher, const Mat& queryDescriptors, const Mat& trainDescriptors, vector<DMatch >& good_matches)
{
	good_matches.clear();
	if (queryDescriptors.rows < thresholdGoodMatches || trainDescriptors.rows < thresholdGoodMatches) {
		return COMPV_ERROR_CODE_S_OK;
	}
#if IMPL_MATCHER == DECL_MATCHER_BRUTE_FORCE_COMPV
	static CompVPtr<CompVArray<uint8_t>* > queryDescriptors_;
	static CompVPtr<CompVArray<uint8_t>* > trainDescriptors_;
	static CompVPtr<CompVArray<CompVDMatch>* > matches_;
	COMPV_CHECK_CODE_RETURN(CompVArray<uint8_t>::newObjStrideless(&queryDescriptors_, (size_t)queryDescriptors.rows, (size_t)queryDescriptors.cols));
	COMPV_CHECK_CODE_RETURN(CompVArray<uint8_t>::newObjStrideless(&trainDescriptors_, (size_t)trainDescriptors.rows, (size_t)trainDescriptors.cols));
	CompVMem::copy((void*)queryDescriptors_->ptr(), queryDescriptors.ptr(0), (queryDescriptors.cols * queryDescriptors.rows));
	CompVMem::copy((void*)trainDescriptors_->ptr(), trainDescriptors.ptr(0), (trainDescriptors.cols * trainDescriptors.rows));
	uint64_t time0 = CompVTime::getNowMills();
	COMPV_CHECK_CODE_RETURN(matcher->process(trainDescriptors_, queryDescriptors_, &matches_));
	uint64_t time1 = CompVTime::getNowMills();
	COMPV_DEBUG_INFO("matcher :%llu millis", (time1 - time0));
#endif

#if IMPL_KNN > 1
#	if IMPL_MATCHER == DECL_MATCHER_BRUTE_FORCE_COMPV
	if (matches_->rows() > 1) {
		const CompVDMatch *match1, *match2;
		int count = min(queryDescriptors.rows - 1, (int)matches_->cols());
		for (int i = 0; i < count; i++) {
			match1 = matches_->ptr(0, i);
			match2 = matches_->ptr(1, i);
			if (match1->distance < ratioTestKNN * match2->distance) {
				good_matches.push_back(DMatch((int)match1->queryIdx, (int)match1->trainIdx, (int)match1->imageIdx, (float)match1->distance));
			}
		}
	}
#	else
	vector<vector<DMatch> > matches;
	matcher->knnMatch(trainDescriptors, queryDescriptors, matches, IMPL_KNN);
	for (int i = 0; i < min(queryDescriptors.rows - 1, (int)matches.size()); i++) {
		if ((matches[i][0].distance < ratioTestKNN*(matches[i][1].distance))) {
			good_matches.push_back(matches[i][0]);
		}
	}
#	endif
#else
#	if IMPL_MATCHER == DECL_MATCHER_BRUTE_FORCE_COMPV
	const CompVDMatch* match;
	for (int i = 0; i < min(queryDescriptors.rows - 1, (int)matches_->cols()); i++) {
		if ((match = matches_->ptr(0, i)) && match->distance <= 35) {
			good_matches.push_back(DMatch(match->queryIdx, match->trainIdx, match->imageIdx, match->distance));
		}
	}
#	else
	vector<DMatch> matches;
	matcher->match(trainDescriptors, queryDescriptors, matches);
	// TODO(dmi): use KNN and "ratio test" (http://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf#page=20)
	for (int i = 0; i < min(queryDescriptors.rows - 1, (int)matches.size()); i++) {
		if (matches[i].distance <= 35.f) {
			good_matches.push_back(matches[i]);
		}
	}
#	endif
#endif

	return COMPV_ERROR_CODE_S_OK;
}

/* itp_point2fToHomogeneous() */
static COMPV_ERROR_CODE itp_point2fToHomogeneous(const vector<Point2f>& cartesian2f, CompVPtrArray(TYPE_COMPV)& homogeneous)
{
	COMPV_CHECK_EXP_RETURN(!cartesian2f.size(), COMPV_ERROR_CODE_E_INVALID_PARAMETER);
	COMPV_CHECK_CODE_RETURN(CompVArray<TYPE_COMPV>::newObjAligned(&homogeneous, 3, cartesian2f.size()));
	TYPE_COMPV* X = const_cast<TYPE_COMPV*>(homogeneous->ptr(0));
	TYPE_COMPV* Y = const_cast<TYPE_COMPV*>(homogeneous->ptr(1));
	TYPE_COMPV* Z = const_cast<TYPE_COMPV*>(homogeneous->ptr(2));
	for (size_t i = 0; i < cartesian2f.size(); ++i) {
		X[i] = cartesian2f[i].x;
		Y[i] = cartesian2f[i].y;
		Z[i] = 1;
	}
	return COMPV_ERROR_CODE_S_OK;
}

/* itp_matToArrayAligned() */
template<int U = TYPE_OPENCV, typename V = TYPE_COMPV>
static COMPV_ERROR_CODE itp_matToArrayAligned(const Mat mat, CompVPtrArray(V)& array)
{
	COMPV_CHECK_CODE_RETURN(CompVArray<V>::copy(array, (const V*)mat.ptr(0), mat.rows, mat.cols));
	return COMPV_ERROR_CODE_S_OK;
}

/* itp_arrayToMat */
template<typename U = TYPE_COMPV, int V = TYPE_OPENCV>
static COMPV_ERROR_CODE itp_arrayToMat(const CompVPtrArray(U)& array, Mat& mat)
{
	mat = Mat((int)array->rows(), (int)array->cols(), V);
	COMPV_CHECK_CODE_RETURN(CompVArray<U>::copy((U*)mat.ptr(0), array));
	return COMPV_ERROR_CODE_S_OK;
}

/* homography() */
static COMPV_ERROR_CODE itp_homography(const vector<Point2f>& srcPoints, const vector<Point2f>& dstPoints, Mat &H)
{
#if IMPL_HOMOGRAPHY == DECL_HOMOGRAPHY_OPENCV
	H = findHomography(srcPoints, dstPoints, CV_RANSAC);
#else
	// Homography 'double' is faster because EigenValues/EigenVectors computation converge faster (less residual error)
	COMPV_ASSERT(srcPoints.size() == dstPoints.size());
	CompVPtrArray(TYPE_COMPV) src_;
	CompVPtrArray(TYPE_COMPV) dst_;
	CompVPtrArray(TYPE_COMPV) h_;
	// CompV requires homogeneous coordinates -> convert from cartesian to homogeneous 2D
	COMPV_CHECK_CODE_RETURN(itp_point2fToHomogeneous(srcPoints, src_));
	COMPV_CHECK_CODE_RETURN(itp_point2fToHomogeneous(dstPoints, dst_));
	COMPV_CHECK_CODE_RETURN(CompVHomography<TYPE_COMPV>::find(src_, dst_, h_));
	COMPV_CHECK_CODE_RETURN((itp_arrayToMat<TYPE_COMPV, TYPE_OPENCV>(h_, H)));
#endif

	return COMPV_ERROR_CODE_S_OK;
}

static COMPV_ERROR_CODE itp_perspectiveTransform(const vector<Point2f>& src, vector<Point2f>& dst, const Mat& m)
{
	COMPV_CHECK_EXP_RETURN(!src.size(), COMPV_ERROR_CODE_E_INVALID_PARAMETER);
	dst.clear();
#if IMPL_PERSPTRANSFORM == DECL_PERSPTRANSFORM_OPENCV
	cv::perspectiveTransform(src, dst, m);
#else
	CompVPtrArray(TYPE_COMPV) src_;
	CompVPtrArray(TYPE_COMPV) dst_;
	CompVPtrArray(TYPE_COMPV) m_;
	COMPV_CHECK_CODE_RETURN(itp_point2fToHomogeneous(src, src_));
	COMPV_CHECK_CODE_RETURN((itp_matToArrayAligned<TYPE_OPENCV, TYPE_COMPV>(m, m_)));
	COMPV_CHECK_CODE_RETURN(CompVTransform<TYPE_COMPV>::perspective2D(src_, m_, dst_));
	const TYPE_COMPV* X = const_cast<TYPE_COMPV*>(dst_->ptr(0));
	const TYPE_COMPV* Y = const_cast<TYPE_COMPV*>(dst_->ptr(1));
	for (size_t i = 0; i < dst_->cols(); ++i) {
		dst.push_back(Point2f((float)X[i], (float)Y[i]));
	}
#endif
	return COMPV_ERROR_CODE_S_OK;
}

static COMPV_ERROR_CODE itp_createCanny(IMPL_CANNY_PTR& canny, float tLow = 0.68f, float tHigh = 0.68*2.f, int32_t kernelSize = 3)
{
#if IMPL_CANNY == DECL_CANNY_COMPV
	COMPV_CHECK_CODE_RETURN(CompVEdgeDete::newObj(COMPV_CANNY_ID, &canny, tLow, tHigh, kernelSize));
#else
	canny.low = tLow;
	canny.high = tHigh;
	canny.kernelSize = kernelSize;
#endif
	return COMPV_ERROR_CODE_S_OK;
}

static COMPV_ERROR_CODE itp_canny(IMPL_CANNY_PTR& canny, const Mat& grayscale, Mat& grad)
{
#if IMPL_CANNY == DECL_CANNY_COMPV
	CompVPtr<CompVImage *> image;
	CompVPtrArray(uint8_t) egdes;
	COMPV_CHECK_CODE_RETURN(CompVImage::wrap(COMPV_PIXEL_FORMAT_GRAYSCALE, grayscale.ptr(0), grayscale.size().width, grayscale.size().height, grayscale.size().width, &image));
	uint64_t time0 = CompVTime::getNowMills();
	COMPV_CHECK_CODE_RETURN(canny->process(image, egdes));
	uint64_t time1 = CompVTime::getNowMills();
	COMPV_DEBUG_INFO("Canny time(CompV): %llu", (time1 - time0));
	grad = Mat(Size((int)egdes->cols(), (int)egdes->rows()), CV_8U);
	for (int j = 0; j < egdes->rows(); ++j) {
		CompVMem::copy(grad.ptr(j), egdes->ptr(j), egdes->rowInBytes());
	}
#else
	uint64_t time0 = CompVTime::getNowMills();
	cv::Scalar mean = cv::mean(grayscale);
	cv::Canny(grayscale, grad, canny.low*mean.val[0], canny.high*mean.val[0], canny.kernelSize);
	uint64_t time1 = CompVTime::getNowMills();
	COMPV_DEBUG_INFO("Canny time(OpenCV): %llu", (time1 - time0));
#endif
	return COMPV_ERROR_CODE_S_OK;
}

static COMPV_ERROR_CODE itp_edges(const Mat& in, Mat& grad, int id = COMPV_SOBEL_ID)
{
	CompVPtr<CompVEdgeDete* > dete;
	CompVPtr<CompVImage *> image;
	CompVPtrArray(uint8_t) egdes;

	COMPV_CHECK_CODE_RETURN(CompVImage::wrap(COMPV_PIXEL_FORMAT_GRAYSCALE, in.ptr(0), in.size().width, in.size().height, in.size().width, &image));
	COMPV_CHECK_CODE_RETURN(CompVEdgeDete::newObj(id, &dete));
	COMPV_CHECK_CODE_RETURN(dete->process(image, egdes));

	grad = Mat(Size((int)egdes->cols(), (int)egdes->rows()), CV_8U);
	for (int j = 0; j < egdes->rows(); ++j) {
		CompVMem::copy(grad.ptr(j), egdes->ptr(j), egdes->rowInBytes());
	}
	return COMPV_ERROR_CODE_S_OK;
}
static COMPV_ERROR_CODE itp_sobel(const Mat& in, Mat& grad) { return itp_edges(in, grad, COMPV_SOBEL_ID); }
static COMPV_ERROR_CODE itp_prewitt(const Mat& in, Mat& grad) { return itp_edges(in, grad, COMPV_PREWITT_ID); }
static COMPV_ERROR_CODE itp_scharr(const Mat& in, Mat& grad) { return itp_edges(in, grad, COMPV_SCHARR_ID); }
static COMPV_ERROR_CODE itp_canny(const Mat& in, Mat& grad, float tLow = 0.66f, float tHigh = 0.66f*2)
{ 
	return itp_edges(in, grad, COMPV_CANNY_ID);
}

static COMPV_ERROR_CODE itp_createHoughStd(IMPL_HOUGHSTD_PTR& houghStd, double rho = 1.f, double theta = kfMathTrigPiOver180, int threshold = 1)
{
#if IMPL_HOUGHSTD == DECL_HOUGHSTD_COMPV
	COMPV_CHECK_CODE_RETURN(CompVHough::newObj(COMPV_HOUGH_STANDARD_ID, &houghStd, (float)rho, (float)theta, (int32_t)threshold));
#else
	houghStd.rho = rho;
	houghStd.theta = theta;
	houghStd.threshold = threshold;
#endif
	return COMPV_ERROR_CODE_S_OK;
}

static COMPV_ERROR_CODE itp_houghStdLines(IMPL_HOUGHSTD_PTR& houghStd, const Mat& in, vector<Vec2f>& lines)
{
	lines.clear();
#if IMPL_HOUGHSTD == DECL_HOUGHSTD_COMPV
	CompVPtrArray(compv_float32x2_t) coords;
	CompVPtrArray(uint8_t) in_;
	COMPV_CHECK_CODE_RETURN((itp_matToArrayAligned<CV_8U, uint8_t>(in, in_)));
	uint64_t time0 = CompVTime::getNowMills();
	COMPV_CHECK_CODE_RETURN(houghStd->process(in_, coords));
	uint64_t time1 = CompVTime::getNowMills();
	COMPV_DEBUG_INFO("HoughLines time(CompV): %llu", (time1 - time0));
	if (coords && !coords->isEmpty()) {
		const compv_float32x2_t* coord = coords->ptr();
		const size_t count = coords->cols();
		for (size_t i = 0; i < count; ++i) {
			lines.push_back(Vec2f(coord[i][0], coord[i][1]));
		}
	}
#else
	uint64_t time0 = CompVTime::getNowMills();
	HoughLines(in, lines, houghStd.rho, houghStd.theta, houghStd.threshold);
	uint64_t time1 = CompVTime::getNowMills();
	COMPV_DEBUG_INFO("HoughLines time(OpenCV): %llu", (time1 - time0));
#endif
	return COMPV_ERROR_CODE_S_OK;
}
