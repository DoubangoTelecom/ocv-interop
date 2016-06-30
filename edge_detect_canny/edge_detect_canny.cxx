#include "../compv_opencv.h"

static COMPV_ERROR_CODE Test0(const Mat& in, Mat& grad_x, Mat& grad_y)
{
	CompVPtr<CompVImage *> image;
	COMPV_CHECK_CODE_RETURN(CompVImage::wrap(COMPV_PIXEL_FORMAT_GRAYSCALE, in.ptr(0), in.size().width, in.size().height, in.size().width, &image));
	static const int16_t ScharrGx_vt[3] = { 3, 10, 3 };
	static const int16_t ScharrGx_hz[3] = { -1, 0, 1 };

	grad_x = Mat(in.size(), CV_16S);
	grad_y = Mat(in.size(), CV_16S);
	int16_t *grad = NULL;
	COMPV_CHECK_CODE_RETURN((CompVMathConvlt::convlt1<uint8_t, int16_t, int16_t>((const uint8_t*)image->getDataPtr(), (size_t)image->getWidth(), (size_t)image->getStride(), (size_t)image->getHeight(), &ScharrGx_vt[0], &ScharrGx_hz[0], (size_t)3, grad)));
	for (int j = 0; j < image->getHeight(); ++j) {
		CompVMem::copy(grad_x.ptr(j), grad + (j * image->getStride()), (image->getWidth() * sizeof(grad[0])));
	}
	COMPV_CHECK_CODE_RETURN((CompVMathConvlt::convlt1<uint8_t, int16_t, int16_t>((const uint8_t*)image->getDataPtr(), (size_t)image->getWidth(), (size_t)image->getStride(), (size_t)image->getHeight(), &ScharrGx_hz[0], &ScharrGx_vt[0], (size_t)3, grad)));
	for (int j = 0; j < image->getHeight(); ++j) {
		CompVMem::copy(grad_y.ptr(j), grad + (j * image->getStride()), (image->getWidth() * sizeof(grad[0])));
	}
	CompVMem::free((void**)&grad);
	return COMPV_ERROR_CODE_S_OK;

}

static COMPV_ERROR_CODE Test1(const Mat& in, Mat& grad)
{
	CompVPtr<CompVImage *> image;
	COMPV_CHECK_CODE_RETURN(CompVImage::wrap(COMPV_PIXEL_FORMAT_GRAYSCALE, in.ptr(0), in.size().width, in.size().height, in.size().width, &image));
	static const int16_t ScharrGx_vt[3] = { 3, 10, 3 };
	static const int16_t ScharrGx_hz[3] = { -1, 0, 1 };
	
	COMPV_ERROR_CODE err = COMPV_ERROR_CODE_S_OK;
	int16_t *gx = NULL, *gy = NULL;
	uint16_t *g = NULL;
	int16_t *gx_, *gy_;
	uint16_t *g_;
	uint16_t max = 1;
	float scale ;
	uint16_t* g16_;
	uint8_t* g8_;

	grad = Mat(in.size(), CV_8U);

	COMPV_CHECK_CODE_BAIL((err = CompVMathConvlt::convlt1<uint8_t, int16_t, int16_t>((const uint8_t*)image->getDataPtr(), (size_t)image->getWidth(), (size_t)image->getStride(), (size_t)image->getHeight(), &ScharrGx_vt[0], &ScharrGx_hz[0], (size_t)3, gx)));
	COMPV_CHECK_CODE_BAIL((err = CompVMathConvlt::convlt1<uint8_t, int16_t, int16_t>((const uint8_t*)image->getDataPtr(), (size_t)image->getWidth(), (size_t)image->getStride(), (size_t)image->getHeight(), &ScharrGx_hz[0], &ScharrGx_vt[0], (size_t)3, gy)));

	g = (uint16_t *)CompVMem::malloc(image->getStride() * image->getHeight() * sizeof(uint16_t));
	COMPV_CHECK_EXP_BAIL(!g, COMPV_ERROR_CODE_E_OUT_OF_MEMORY);

	gx_ = gx; 
	gy_ = gy;
	g_ = g;

	// compute gradient and find max
	for (int j = 0; j < image->getHeight(); ++j) {
		for (int i = 0; i < image->getWidth(); ++i) {
			g_[i] = abs(gx_[i]) + abs(gy_[i]);
			//g_[i] = (uint16_t)sqrt((gx_[i] * gx_[i]) + (gy_[i] * gy_[i]));
			if (max < g_[i]) {
				max = g_[i];
			}
		}
		g_ += image->getStride();
		gx_ += image->getStride();
		gy_ += image->getStride();
	}

	// scale (normalization)
	scale = 255.f / float(max);
	g16_ = g;
	g8_ = (uint8_t*)g;
	for (int j = 0; j < image->getHeight(); ++j) {
		for (int i = 0; i < image->getWidth(); ++i) {
			g8_[i] = (uint8_t)COMPV_MATH_CLIP3(0, 255, (g16_[i] * scale));
		}
		g16_ += image->getStride();
		g8_ += image->getStride();
	}

	// copy
	g8_ = (uint8_t*)g;
	for (int j = 0; j < image->getHeight(); ++j) {
		CompVMem::copy(grad.ptr(j), g8_ + (j * image->getStride()), (image->getWidth() * sizeof(g8_[0])));
	}

bail:
	CompVMem::free((void**)&gx);
	CompVMem::free((void**)&gy);
	CompVMem::free((void**)&g);
	return err;
}

static COMPV_ERROR_CODE Test2(const Mat& in, Mat& grad)
{
	CompVPtr<CompVEdgeDete* > dete;
	CompVPtr<CompVImage *> image;
	CompVPtrArray(uint8_t) egdes;

	COMPV_CHECK_CODE_RETURN(CompVImage::wrap(COMPV_PIXEL_FORMAT_GRAYSCALE, in.ptr(0), in.size().width, in.size().height, in.size().width, &image));
	COMPV_CHECK_CODE_RETURN(CompVEdgeDete::newObj(COMPV_SCHARR_ID, &dete));
	COMPV_CHECK_CODE_RETURN(dete->process(image, egdes));

	grad = Mat(Size((int)egdes->cols(), (int)egdes->rows()), CV_8U);
	for (int j = 0; j < egdes->rows(); ++j) {
		CompVMem::copy(grad.ptr(j), egdes->ptr(j), egdes->rowInBytes());
	}
	return COMPV_ERROR_CODE_S_OK;
}

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
	IMPL_DETECTOR_PTR detector = NULL;

	COMPV_CHECK_CODE_ASSERT(itp_createDetector(detector));

	cv::VideoCapture cap(IMPL_CAMERA_ID);
	cap.set(CV_CAP_PROP_FPS, IMPL_CAMERA_FRAME_RATE);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, IMPL_CAMERA_FRAME_WIDTH);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, IMPL_CAMERA_FRAME_HEIGHT);
	COMPV_ASSERT(cap.isOpened());

	do {
		cap >> frame;
		//frame = imread("C:/Projects/GitHub/compv/tests/Bikesgray.jpg");

		//GaussianBlur(frame, frame, Size(3, 3), 0, 0, BORDER_DEFAULT);
		COMPV_CHECK_CODE_ASSERT(itp_imageBgrToGrayscale(frame, src_gray));
		
		Mat grad_x, grad_y;
		Mat abs_grad_x, abs_grad_y, grad;

		uint64_t time0 = CompVTime::getNowMills();

#if 0
		Test1(src_gray, grad);
#elif 1
		Test2(src_gray, grad);
#elif 0
		Test0(src_gray, grad_x, grad_y);
		convertScaleAbs(grad_x, abs_grad_x);
		convertScaleAbs(grad_y, abs_grad_y);
		addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
#elif 0
		/// Gradient X
		Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
		//Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);

		/// Gradient Y
		Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
		//Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(grad_y, abs_grad_y);

		/// Total Gradient (approximate)
		addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
#else
		Canny(src_gray, grad, 170, 90, 3);
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

