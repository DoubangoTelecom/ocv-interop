#include <compv/compv_api.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#if COMPV_OS_WINDOWS
#	include <tchar.h>
#endif

using namespace cv;
using namespace compv;
using namespace std;

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
#define IMPL_MAX_FEATURES				2000 // 2000 // default is 500 but this dosen't provide good result (OpenCV default value: 500)
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
#elif IMPL_PRESET == DECL_PRESET_COMPV
#	define IMPL_DETECTOR		DECL_DETECTOR_ORB_COMPV
#	define IMPL_EXTRACTOR		DECL_EXTRACTOR_ORB_COMPV
#	define IMPL_MATCHER			DECL_MATCHER_BRUTE_FORCE_COMPV
#	define IMPL_HOMOGRAPHY		DECL_HOMOGRAPHY_COMPV
#	define IMPL_PERSPTRANSFORM	DECL_PERSPTRANSFORM_COMPV
#elif IMPL_PRESET == DECL_PRESET_SURF
#	define IMPL_DETECTOR		DECL_DETECTOR_SURF
#	define IMPL_EXTRACTOR		DECL_EXTRACTOR_SURF
#	define IMPL_MATCHER			DECL_MATCHER_BRUTE_FORCE
#	define IMPL_HOMOGRAPHY		DECL_HOMOGRAPHY_OPENCV
#	define IMPL_PERSPTRANSFORM	DECL_PERSPTRANSFORM_OPENCV
#elif IMPL_PRESET == DECL_PRESET_SIFT
#	define IMPL_DETECTOR		DECL_DETECTOR_SIFT
#	define IMPL_EXTRACTOR		DECL_EXTRACTOR_SIFT
#	define IMPL_MATCHER			DECL_MATCHER_BRUTE_FORCE
#	define IMPL_HOMOGRAPHY		DECL_HOMOGRAPHY_OPENCV
#	define IMPL_PERSPTRANSFORM	DECL_PERSPTRANSFORM_OPENCV
#else // NONE
#	define IMPL_DETECTOR		DECL_DETECTOR_ORB_COMPV
#	define IMPL_EXTRACTOR		DECL_EXTRACTOR_ORB_COMPV
#	define IMPL_MATCHER			DECL_MATCHER_BRUTE_FORCE_COMPV
#	define IMPL_HOMOGRAPHY		DECL_HOMOGRAPHY_COMPV
#	define IMPL_PERSPTRANSFORM	DECL_PERSPTRANSFORM_COMPV
#endif

#if IMPL_DETECTOR == DECL_DETECTOR_ORB_COMPV
#	define IMPL_DETECTOR_PTR CompVPtr<CompVFeatureDete* >
#else
#	define IMPL_DETECTOR_PTR cv::Ptr<FeatureDetector>
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
static COMPV_ERROR_CODE itp_createDetector(IMPL_DETECTOR_PTR& detector)
{
#if IMPL_DETECTOR == DECL_DETECTOR_ORB_COMPV
	float scaleFactor = 0.83f; // (1 / 1.2)
	int32_t nlevels = IMPL_PYRAMID_LEVELS;
	bool nms = IMPL_NMS;
	int32_t fastThreshold = IMPL_FAST_THRESHOLD;
	int nfeatures = IMPL_MAX_FEATURES;
	COMPV_CHECK_CODE_RETURN(CompVFeatureDete::newObj(COMPV_ORB_ID, &detector));
	COMPV_CHECK_CODE_RETURN(detector->set(COMPV_ORB_SET_INT32_FAST_THRESHOLD, &fastThreshold, sizeof(fastThreshold)));
	COMPV_CHECK_CODE_RETURN(detector->set(COMPV_ORB_SET_BOOL_FAST_NON_MAXIMA_SUPP, &nms, sizeof(nms)));
	COMPV_CHECK_CODE_RETURN(detector->set(COMPV_ORB_SET_INT32_PYRAMID_LEVELS, &nlevels, sizeof(nlevels)));
	COMPV_CHECK_CODE_RETURN(detector->set(COMPV_ORB_SET_FLOAT_PYRAMID_SCALE_FACTOR, &scaleFactor, sizeof(scaleFactor)));
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
	COMPV_CHECK_CODE_ASSERT(detector->process(queryImage, interestPoints));
	for (size_t i = 0; i < interestPoints->size(); ++i) {
		const CompVInterestPoint* p = interestPoints->ptr(i);
		keypoints.push_back(KeyPoint(p->x, p->y, p->size, (float)p->orient, (float)p->strength, p->level));
	}
#else
	detector->detect(grayImage, keypoints);
#endif

	return COMPV_ERROR_CODE_S_OK;
}