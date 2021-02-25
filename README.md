# opencv-segmenting-water-using-2-layer-MLP-RGB-feature


OpenCV work demonstrating two hidden layer perceptron NN trained with RGB intensity values as feature vector for segmenting water pixels from non-water pixel in underwater images

Images acquired in underwater environments undergo a degradation process due to the inherent complexity of the interaction of the light with the medium. Such interaction includes numerous phenomena such as multipath refraction and reflection of light rays on particles in suspension with dynamic motion patterns.


An increasing number of real-world applications are related to underwater environments. These applications include Fisheries, environmental and structural monitoring inspections and Oil and Gas exploration.


Techniques based on machine vision can help humans to monitor and to supervise activities in these scenarios, as well as to enable carrying out missions with autonomous robotic vehicles.

The following work using OpenCV (4.2) library uses multi-layer (four) perceptron neural network for train and prediction of water pixels in underwater images while considering Red, Green and Blue intensity values as the feature vector. This work also gives the chance of examining the effect of varying its #neurons under thousand neurons. Note that, it may not keep on working for more than a thousand neurons.
This work demonstrates using NN for classification of 3x3 patches extracted by dividing the entire image into 3x3 patches. Therefore, the NN predicts for each patch separately. 
The accuracy is calculated by accumulating all the NN patch responses for water class (2nd neuron output) and finally dividing this accumulation by the number of actual water pixels (409259 pixels). 
TrainData:
It holds the training data that must be CV_32FC1 type and its dimension is nSamples * nFeatures
TrainLabels:
It holds the labels for each sample that must be CV_32FC1 type and its dimension is nSamples * 1
TestLabels:
It holds the correct labels for test data and it is also used in accuracy measuring

The first part of the code deals with red channel compensation of the image which improves the bluish color bias of the image. For red channel compensation, equation 8 of paper "Color Correction Based on CFA and Enhancement Based on Retinex With Dense Pixels for Underwater Images" have been used. The red channel compensated image is stored in image2 in the code. 

Pure blue water samples are loaded as two training image samples (two .png files). These pure water samples are named sample (1).png and sample (2).png. They are labeled +1.0 in the training label vector.

Non water samples are chosen within another input image using this rectangle window (1, 400, 640, 320). They are labeled 0.0 in the non-water training label vector.

Parameters in ANN_MLP:

-	Sigmoid activation function
-	300 iteration
-	1e-4 as max error
-	The feature vector has three columns holding R-G-B intensities respectively (e.g. mx3).

Image Data Base:
you can use your own training images. This code is using two training images named sample (1).png and sample (2).png which are pure water pixels. These two files take label 1.0.
The input image is an underwater image from this dataset:
  ( https://li-chongyi.github.io/proj_benchmark.html ). 

Conclusion:

In brief, the code's purpose is to demonstrate the performance of Red, Green and Blue intensity values in segmenting water pixel regions from non-water regions using two hidden layer perceptron ANN with a specified layer structure which can be changed if you need. 
Since our results may be inaccurate in case of different systems, the plot of accuracy vs #neurons is not guaranteed to be %100 accurate.
In this case, we insist on replicating the process by yourself. 


















The following code is function main() + 
Train_test_ANN_MLP_SIGMOID_BACKPROP_PATCH:



	#include <iostream>
	#include <opencv2/core/core.hpp>
	#include <opencv2/highgui/highgui.hpp>
	#include <opencv2/imgproc/imgproc.hpp>
	#include <opencv2/imgcodecs.hpp>
	#include <opencv2/ml/ml.hpp>
	#include "underwater.h"
	#include <fstream>
	using namespace cv;
	using namespace cv::ml;
	using namespace std;

	Mat image;

	void Train_test_ANN_MLP_SIGMOID_BACKPROP_PATCH(
	    int nclasses, const Mat& TrainData, const Mat& TrainLabels, Mat& TestLabels) {
	    //TrainData = ntrainsamples * nFeatures - float typr
	    //TrainLabels=ntrainsamples*1   - float type
	    //TestData=ntestsamples*nfeatures - float type

    // print function is starting
    std::printf("\n\nTrain_Test_ANN_MLP_SIGMOID_BackProp\n");

    // ann requires "one-hot" encoding of class labels:
    Mat TrainClasses = Mat::zeros(TrainData.rows, nclasses, CV_32FC1);
    for (int counterLabel = 0; counterLabel < TrainClasses.rows; counterLabel++) {
        TrainClasses.at<float>(counterLabel, (int)(TrainLabels.at<int>(counterLabel))) = 1.f;
    }// end for counterLabel

    // convert test label to one-hot encoded
    Mat TestClass = Mat::zeros(TestLabels.rows, nclasses, CV_32FC1);
    //printminMaxLoc(TestLabels);

    for (int i = 0; i < TestClass.rows; i++) {
        TestClass.at<float>(i, (int)(TestLabels.at<float>(i))) = 1.f;
    }// end for i


    // size of the square patch
    int GridSize = 3;

    //// hold every patch from input image
    Mat InputCellImage;
    // column vector version of inputcell image
    Mat InputCellVector_float;
    // hold every patch from prediction result
    Mat OutputCellImage;
    // hold responses of all patches
    Mat ResponseImage;
    ResponseImage.create(image.size(), CV_32FC2);
    // image dims:
    int nRowsImage = image.rows;//height
    int nColsImage = image.cols;//width

    int nNeuron = 28;
    int nNeuron2 = 28;
    int nFeatures = TrainData.cols;
    // define the network pointer
    Ptr<cv::ml::ANN_MLP> ANN_MLP_SIGMOID_BackProp = ml::ANN_MLP::create();
    // layers holds details of network
    Mat_<int> ANNlayers(4, 1);
    ANNlayers(0) = nFeatures; // input
    ANNlayers(1) = nNeuron; //hidden 1st
    ANNlayers(2) = nNeuron2; // hidden 2nd
    ANNlayers(3) = nclasses; // output, 1 pin per class
    ANN_MLP_SIGMOID_BackProp->setLayerSizes(ANNlayers);
    ANN_MLP_SIGMOID_BackProp->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM, 0, 0);
    ANN_MLP_SIGMOID_BackProp->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 300, 0.0001));
    ANN_MLP_SIGMOID_BackProp->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, 0.0001);

    // time measurment 
    double t;

    cerr << TrainData.size() << "" << TrainClasses.size() << endl;
    t = (double)getTickCount();
    ANN_MLP_SIGMOID_BackProp->train(TrainData, cv::ml::ROW_SAMPLE, TrainClasses);
    t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "Train Time seconds:\t\t" << t << endl;

    // sum of prediction vector on water class
    float SumPredictionVector = 0.0f;
    // loop for every patch
    for (int y = 0; y < nRowsImage - GridSize; y += GridSize) {
        for (int x = 0; x < nColsImage - GridSize; x += GridSize) {
            //index address of current pixel topleft of a patch
            int k = x * y + x;
            // rect holding patch rect
            Rect CellRect(x, y, GridSize, GridSize);
            // get this patch
            InputCellImage = image(CellRect);
            // convert this patch to float
            InputCellImage.convertTo(InputCellImage, CV_32FC3);

            // reshape cell image from 3 planes to a 3 column
            // vector and convert to float
            InputCellImage.reshape(1, InputCellImage.rows * InputCellImage.cols).convertTo(InputCellVector_float, CV_32FC1);
            // mat for predictions of ANN - float type- (Gridsize*gridsize)*2
            Mat PredictionVector;

            //t = (double)getTickCount();
            ANN_MLP_SIGMOID_BackProp->predict(InputCellVector_float, PredictionVector);
            //t = ((double)getTickCount() - t) / getTickFrequency();
            //cout << "Time Prediction seconds\t\t" << t << endl;
            SumPredictionVector += cv::sum(PredictionVector.col(1))[0];
            
            //// reshape prediction vector to a square
            OutputCellImage = PredictionVector.reshape(2, GridSize);

            // put this patch response into a whole image(= image)
            ResponseImage(CellRect) = OutputCellImage;
            
        }// // end for x
    }// end for y
    float nWaterPixels = 409259.0f;
    std::cout << "water accuracy:\t\t" << SumPredictionVector / nWaterPixels << endl;

	int main()
	{

    //read an image
    image = imread("water (8).png", 1);

    //check for existance of data
    if (!image.data)
    {
        printf("no image data.\n"); return -1;
    }

    if (image.isContinuous()) {
        cout << "loaded image is continuous" << "\n";
    }
    // hold HSV conversion of image
    Mat HSVimage;
    // convert to hsv - 0 <  hsvimage < 255
    cv::cvtColor(image, HSVimage, cv::COLOR_BGR2HSV);
    double min, max;
    //cv::minMaxLoc(HSVimage, &min, &max);
    //std::cout << "min and max:\t" << min << "\t" << max << "\n";

        
    FileStorage PrintInXMLFile("results.xml", FileStorage::WRITE);
    

    //int threshold_value = 250;
    //int threshold_type = 0;// 0 is binary
    //int max_binary_value = 255;


    Mat RedCompensatedImage;
    RedChannelCompensationEq8(image, 0.05f, RedCompensatedImage);

    // getting water feature vector
    Mat waterFeatureVectorRGB_uchar, waterLabelVector_float;
    createWaterOnlyFeatureVector(waterFeatureVectorRGB_uchar, waterLabelVector_float);

    // get nonwater feature vector
    Mat nonwaterFeatureVectorRGB_uchar, nonwaterLabelVector_float;
    CreateNonWaterFeatureVector(nonwaterFeatureVectorRGB_uchar, nonwaterLabelVector_float);

    // label matrix for all data
    Mat AllLabelVector_float;
    cv::vconcat(waterLabelVector_float, nonwaterLabelVector_float, AllLabelVector_float);
    std::printf("AllLabelVector_float dims:  %d\t%d\t%d\t\n", AllLabelVector_float.rows, AllLabelVector_float.cols, AllLabelVector_float.channels());

    //concatenation of all training data
    // uchar mat
    Mat AllTrainingDataVector_uchar;
    cv::vconcat(waterFeatureVectorRGB_uchar,nonwaterFeatureVectorRGB_uchar, AllTrainingDataVector_uchar);
    // convert to double
    Mat AllTrainingDataVector_float;
    AllTrainingDataVector_uchar.convertTo(AllTrainingDataVector_float, CV_32F);
    std::printf("AllTrainingDataVector_float dims: %d\t%d\t%d\t\n", AllTrainingDataVector_float.rows, AllTrainingDataVector_float.cols, AllTrainingDataVector_float.channels());
    //std::cout << AllTrainingDataVector_float << "\n";

    // reshape the test image
    Mat TestDataColumnVector_float;
    image.reshape(1, image.rows * image.cols).convertTo(TestDataColumnVector_float, CV_32FC1);

    Mat PreLabeledimage = (imread("binary image.png", 0) > 0)/255.0;
    Mat PreLabeledVector_float;
    PreLabeledimage.reshape(1, PreLabeledimage.rows * PreLabeledimage.cols).convertTo(PreLabeledVector_float, CV_32FC1);
    cv::minMaxLoc(PreLabeledVector_float, &min, &max);
    std::cout << "min and max PreLabeledVector_float:\t" << min << "\t" << max << "\n";
    Train_test_ANN_MLP_SIGMOID_BACKPROP_PATCH(2, AllTrainingDataVector_float, AllLabelVector_float, PreLabeledVector_float);

    PrintInXMLFile.release();//release the file after writing

    cv::waitKey(0);

    return 0;

	}// end main








Here is the code for underwater.h:





	#include <iostream>
	#include <opencv2/core/core.hpp>
	#include <opencv2/highgui/highgui.hpp>
	#include <opencv2/imgproc/imgproc.hpp>
	#include <opencv2/imgcodecs.hpp>
	#include <opencv2/ml/ml.hpp>
	using namespace cv;
	using namespace cv::ml;
	using namespace std;


	constexpr auto trainingsampleaddress = "sample (%d).png";
	//create an Mx3 vector of RGB values
	int createWaterOnlyFeatureVector(Mat &waterTrainingFeatureVectorRGB_uchar, Mat &waterTrainingLabelVector_float) {

	std::printf("\n\n function createWaterOnlyFeatureVector\n\n");
	// char array to store filenames
	char waterTrainingFileName[128];

	// num of available pure water sample images
	uint waterTrainingTotalSamples = 2;

	// holding all rgb pixels
	Mat waterTrainingRGBVector;

	// holding all labels
	Mat waterTrainingLabelsVector_float;
	
	for (uint i = 1; i <= waterTrainingTotalSamples; i++) {

		// show the training image number in output
		std::printf("\nimage training #%d\n", i);
		//creating filenames				
		std::snprintf(waterTrainingFileName, sizeof(waterTrainingFileName), trainingsampleaddress, i);

		//read image
		Mat waterTrainingImage = imread(waterTrainingFileName, 1);

		//check for data reading to be ok
		if (!waterTrainingImage.data) {
			std::printf("failed to fetch file %s\n", waterTrainingFileName);
			return -1;
		}//end of if 
		//imshow("water only training", waterTrainingImage);
		//waitKey(0);

		Mat WaterOnlyPlanes[3];
		split(waterTrainingImage, WaterOnlyPlanes);
		std::printf("size of WaterOnlyPlanes[0]: %d  %d  %d\n", WaterOnlyPlanes[0].rows, WaterOnlyPlanes[0].cols, WaterOnlyPlanes[0].channels());

		// vector of each plane
		Mat WaterOnlyRedVector, WaterOnlyGreenVector, WaterOnlyBlueVector;

		// reshape each plane into vector separately
		// column vector
		WaterOnlyRedVector = WaterOnlyPlanes[2].reshape(0, 1).t();
		WaterOnlyGreenVector = WaterOnlyPlanes[1].reshape(0, 1).t();
		WaterOnlyBlueVector = WaterOnlyPlanes[0].reshape(0, 1).t();
		std::printf("WaterOnlyRedVector size: %d  %d  %d\n", WaterOnlyRedVector.rows, WaterOnlyRedVector.cols, WaterOnlyRedVector.channels());

		// (row*com)x3 vector holding all RGB pixels of water
		// (WaterOnlyRedVector.rows, 3);
		Mat WaterOnlyRGBVector;

		// concatenation of three vectors values using hconcat
		// hconcate: horizontal concatenation
		hconcat(WaterOnlyRedVector, WaterOnlyGreenVector, WaterOnlyRGBVector);
		hconcat(WaterOnlyRGBVector, WaterOnlyBlueVector, WaterOnlyRGBVector);
		std::printf("WaterOnlyRGBVector dims is: %d  %d  %d\n", WaterOnlyRGBVector.rows, WaterOnlyRGBVector.cols, WaterOnlyRGBVector.channels());

		// label vector for water pixels, all preset one 
		Mat WaterOnlyLabelVector_float = Mat::ones(WaterOnlyRGBVector.rows, 1, CV_32SC1);
		std::printf("WaterOnlyLabelVector_float dims: %d\t%d\t%d\t\n", WaterOnlyLabelVector_float.rows, WaterOnlyLabelVector_float.cols, WaterOnlyLabelVector_float.channels());
		//assign value to 
		if (!waterTrainingRGBVector.data) {
			//if this the first time assigning
			// no need to vconcat
			waterTrainingRGBVector = WaterOnlyRGBVector;
			std::printf("dims of waterTrainingRGBVector: %d\t%d\t%d\t\n", waterTrainingRGBVector.rows, waterTrainingRGBVector.cols, waterTrainingRGBVector.channels());
		}//end of if
		else {
			cv::vconcat(waterTrainingRGBVector, WaterOnlyRGBVector, waterTrainingRGBVector);
			std::printf("dims of waterTrainingRGBVector: %d\t%d\t%d\t\n", waterTrainingRGBVector.rows, waterTrainingRGBVector.cols, waterTrainingRGBVector.channels());
		}//end of else

		//////////////////////////////
		//labeling
		// for label +1, blue water label
		if (i <= 6) {
			if (!waterTrainingLabelsVector_float.data) {
				waterTrainingLabelsVector_float = WaterOnlyLabelVector_float;
				std::printf("waterTrainingLabelsVector_float dims: %d\t%d\t%d\t\n\n", waterTrainingLabelsVector_float.rows, waterTrainingLabelsVector_float.cols, waterTrainingLabelsVector_float.channels());
			}
			else {
				cv::vconcat(waterTrainingLabelsVector_float, WaterOnlyLabelVector_float, waterTrainingLabelsVector_float);
				std::printf("waterTrainingLabelsVector_float dims: %d\t%d\t%d\t\n\n", waterTrainingLabelsVector_float.rows, waterTrainingLabelsVector_float.cols, waterTrainingLabelsVector_float.channels());
			} // end of else
		} // end if i<=6
		// label +2.0 for green water
		else if(i > 6 && i <= 11) {
			WaterOnlyLabelVector_float += 1.0;
			cv::vconcat(waterTrainingLabelsVector_float, WaterOnlyLabelVector_float, waterTrainingLabelsVector_float);
			std::printf("waterTrainingLabelsVector_float dims: %d\t%d\t%d\t\n\n", waterTrainingLabelsVector_float.rows, waterTrainingLabelsVector_float.cols, waterTrainingLabelsVector_float.channels());
		}
		if (i == 12) {
			WaterOnlyLabelVector_float += 2.0;
			cv::vconcat(waterTrainingLabelsVector_float, WaterOnlyLabelVector_float, waterTrainingLabelsVector_float);
			std::printf("waterTrainingLabelsVector_float dims: %d\t%d\t%d\t\n\n", waterTrainingLabelsVector_float.rows, waterTrainingLabelsVector_float.cols, waterTrainingLabelsVector_float.channels());
		}//end if i==12
		if (i == 13) {
			WaterOnlyLabelVector_float += 3.0;
			cv::vconcat(waterTrainingLabelsVector_float, WaterOnlyLabelVector_float, waterTrainingLabelsVector_float);
			std::printf("waterTrainingLabelsVector_float dims: %d\t%d\t%d\t\n\n", waterTrainingLabelsVector_float.rows, waterTrainingLabelsVector_float.cols, waterTrainingLabelsVector_float.channels());
		}//end if i==13
		if (i == 14 || i == 15) {			
			WaterOnlyLabelVector_float += 4.0;
			cv::vconcat(waterTrainingLabelsVector_float, WaterOnlyLabelVector_float, waterTrainingLabelsVector_float);
			std::printf("waterTrainingLabelsVector_float dims: %d\t%d\t%d\t\n\n", waterTrainingLabelsVector_float.rows, waterTrainingLabelsVector_float.cols, waterTrainingLabelsVector_float.channels());
		}//end if i==14 || i==15


	}//end of for reading water training samples

	// assign the output
	waterTrainingFeatureVectorRGB_uchar = waterTrainingRGBVector;
	waterTrainingLabelVector_float = waterTrainingLabelsVector_float;

	}// end of function createWaterOnlyFeatureVector


	int CreateNonWaterFeatureVector(Mat &NonWaterFetureVectorRGB_uchar, Mat &NonWaterLabelVector_float) {

	//read an image
	Mat image = imread("9554.png", 1);
	//check for existance of data
	if (!image.data)
	{
		std::printf("no image data.\n"); return -1;
	}

	// defining non water coordination
	Rect nonWaterRect(1, 400, 640, 320);
	//non water image
	Mat NonWaterImage = image(nonWaterRect);
	//imshow("non water image", NonWaterImage);

	//holding plited nonwater image planes
	Mat NonWaterPlanes[3];

	//split nonwater image sample
	split(NonWaterImage, NonWaterPlanes);
	std::printf("NonWaterPlanes[0] dims: %d\t%d\t%d \n", NonWaterPlanes[0].rows, NonWaterPlanes[0].cols, NonWaterPlanes[0].channels());

	// 3 column vector for each of rgb planes
	Mat NonWaterRedVector, NonWaterGreenVector, NonWaterBlueVector;

	// reshaping each plane to get a column vector Mx1
	NonWaterRedVector = NonWaterPlanes[2].reshape(0, 1).t();//red 1st
	NonWaterGreenVector = NonWaterPlanes[1].reshape(0, 1).t();//green 2nd
	NonWaterBlueVector = NonWaterPlanes[0].reshape(0, 1).t();//blue
	std::printf("NonWaterGreenVector size is: %d\t%d\t%d\t\n", NonWaterGreenVector.rows, NonWaterGreenVector.cols, NonWaterGreenVector.channels());

	// Mx3 vector holding all RGB pixels of NonWater
	Mat NonWaterRGBVector;

	// concatenate 3 column vec into one place (Mx3)
	cv::hconcat(NonWaterRedVector, NonWaterGreenVector, NonWaterRGBVector);
	cv::hconcat(NonWaterRGBVector, NonWaterBlueVector, NonWaterRGBVector);
	std::printf("NonWaterRGBVector dims: %d\t%d\t%d\t\n", NonWaterRGBVector.rows, NonWaterRGBVector.cols, NonWaterRGBVector.channels());

	// label vector for NonWater pixels
	NonWaterLabelVector_float = cv::Mat::zeros(NonWaterRGBVector.rows, 1, CV_32SC1);
	std::printf("NonWaterLabelVector_float dims: %d\t%d\t%d\t\n", NonWaterLabelVector_float.rows, NonWaterLabelVector_float.cols, NonWaterLabelVector_float.channels());
	//cout << NonWaterLabelVector_float << "\n";
	//cout << NonWaterLabelVector_float << "\n";

	// assign output
		NonWaterFetureVectorRGB_uchar = NonWaterRGBVector;
	}// end of function createWaterOnlyFeatureVector


	////////////////////////////////////////
	int RedChannelCompensationEq8(Mat& image, float AlphaCoefficient, Mat &RedCompensatedImage) {
		// red channel compensation based on eq.8 
		// of paper Color Correction Based on CFA and Enhancement
		//Based on Retinex With Dense Pixels for
		//Underwater Images

	
	//planes is a vector for holding rgb channels separately
	//std::vector<Mat> planes;
	Mat planes[3];

	//split the image into channels
	//planes[2] is the red channel
	split(image, planes);
	//cout<<cv::min(planes[0],2.0f)
	// converting planes from uchar to double
	planes[0].convertTo(planes[0], CV_64FC1);
	planes[1].convertTo(planes[1], CV_64FC1);
	planes[2].convertTo(planes[2], CV_64FC1);

	// defining coefficients of green and blue channel for blending
	double a = AlphaCoefficient, b = 1.0 - a;

	//sum_im stores pixelwise sum of Red, Green and Blue planes
	Mat imBlendNormal_B_G, sum_im;

	//converting to double
	imBlendNormal_B_G.convertTo(imBlendNormal_B_G, CV_64FC1);
	sum_im.convertTo(sum_im, CV_64FC1);

	//blending green and blue planes with a and b coefficients
	// and 0.0 offset(or gamma)
	addWeighted(planes[1], a, planes[0], b, 0.0, imBlendNormal_B_G);

	// sum of red, green and blue pixel in two addWeighted calls
	addWeighted(planes[2], 1.0, planes[1], 1.0, 0.0, sum_im);
	addWeighted(planes[0], 1.0, sum_im, 1.0, 0.0, sum_im);

	//dividing blended green and blue image to total RGB sum
	divide(imBlendNormal_B_G, sum_im, imBlendNormal_B_G);

	//defining average kernel 3x3
	Mat avg3x3_kernel = (Mat_<double>(3, 3) << 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0);

	//defining matrices for storing 3x3 average of blue and green planes
	Mat blueAverage, greenAverage;
	// converting to double type
	blueAverage.convertTo(blueAverage, CV_64FC1);
	greenAverage.convertTo(greenAverage, CV_64FC1);

	// taking 3x3 average
	filter2D(planes[0], blueAverage, planes[0].depth(), avg3x3_kernel);
	filter2D(planes[1], greenAverage, planes[1].depth(), avg3x3_kernel);

	//imBlendAverage_B_G_R: for blending of averaged green and blue channels
	Mat imBlendAverage_B_G_R;
	//convert to double
	imBlendAverage_B_G_R.convertTo(imBlendAverage_B_G_R, CV_64FC1);

	//blend averaged green and blue with a and b coeffs
	addWeighted(greenAverage, a, blueAverage, b, 0.0, imBlendAverage_B_G_R);

	//differentiate red values
	addWeighted(imBlendAverage_B_G_R, 1.0, planes[2], -1.0, 0.0, imBlendAverage_B_G_R);

	//CompensationTermRed: storing finally compensated red channel intensities
	Mat CompensationTermRed;
	//coverting to double
	CompensationTermRed.convertTo(CompensationTermRed, CV_64FC1);

	//multiplication term
	CompensationTermRed = imBlendAverage_B_G_R.mul(imBlendNormal_B_G);

	//final add term
	addWeighted(CompensationTermRed, 1.0, planes[2], 1.0, 0.0, CompensationTermRed);

	// assign new red channel values to planes[2]
	planes[2] = CompensationTermRed;

	Mat image2;
	cv::merge(planes, 3, image2);
	image2.convertTo(image2, CV_8UC3);
	/*imshow("merge", image2);
	std::printf("\ndims of image2 (merge): %d  %d\n", image2.rows, image2.cols);*/

	// assign output
	RedCompensatedImage = image2;

	return 0;

	}//end of function RedChannelCompensationEq8

	void imshowwindownormal(const char name4window[128], Mat& tmp4image) {
		namedWindow(name4window, cv::WINDOW_NORMAL); 
		imshow(name4window, tmp4image);
	}



	int createWaterOnlyHSVFeatureVector(Mat& waterTrainingFeatureVectorRGB_uchar, Mat& waterTrainingLabelVector_float) {

	std::printf("\n\n function createWaterOnlyFeatureVector\n\n");
	// char array to store filenames
	char waterTrainingFileName[128];

	// num of available pure water sample images
	uint waterTrainingTotalSamples = 2;

	// holding all rgb pixels
	Mat waterTrainingRGBVector;

	// holding all labels
	Mat waterTrainingLabelsVector_float;

	for (uint i = 1; i <= waterTrainingTotalSamples; i++) {

		// show the training image number in output
		std::printf("\nimage training #%d\n", i);
		//creating filenames				
		std::snprintf(waterTrainingFileName, sizeof(waterTrainingFileName), trainingsampleaddress, i);

		//read image
		Mat waterTrainingImage = imread(waterTrainingFileName, 1);

		// convert to hsv
		cv::cvtColor(waterTrainingImage, waterTrainingImage, cv::COLOR_BGR2HSV);

		//check for data reading to be ok
		if (!waterTrainingImage.data) {
			std::printf("failed to fetch file %s\n", waterTrainingFileName);
			return -1;
		}//end of if 
		//imshow("water only training", waterTrainingImage);
		//waitKey(0);

		Mat WaterOnlyPlanes[3];
		split(waterTrainingImage, WaterOnlyPlanes);
		std::printf("size of WaterOnlyPlanes[0]: %d  %d  %d\n", WaterOnlyPlanes[0].rows, WaterOnlyPlanes[0].cols, WaterOnlyPlanes[0].channels());

		// vector of each plane
		Mat WaterOnlyValueVector, WaterOnlySaturationVector, WaterOnlyHueVector;

		// reshape each plane into vector separately
		// column vector
		WaterOnlyValueVector = WaterOnlyPlanes[2].reshape(0, 1).t();
		WaterOnlySaturationVector = WaterOnlyPlanes[1].reshape(0, 1).t();
		WaterOnlyHueVector = WaterOnlyPlanes[0].reshape(0, 1).t();
		std::printf("WaterOnlyValueVector size: %d  %d  %d\n", WaterOnlyValueVector.rows, WaterOnlyValueVector.cols, WaterOnlyValueVector.channels());

		// (row*com)x3 vector holding all RGB pixels of water
		// (WaterOnlyValueVector.rows, 3);
		Mat WaterOnlyRGBVector;

		// concatenation of three vectors values using hconcat
		// hconcate: horizontal concatenation
		hconcat(WaterOnlyValueVector, WaterOnlySaturationVector, WaterOnlyRGBVector);
		hconcat(WaterOnlyRGBVector, WaterOnlyHueVector, WaterOnlyRGBVector);
		std::printf("WaterOnlyRGBVector dims is: %d  %d  %d\n", WaterOnlyRGBVector.rows, WaterOnlyRGBVector.cols, WaterOnlyRGBVector.channels());

		// label vector for water pixels, all preset one 
		Mat WaterOnlyLabelVector_float = Mat::ones(WaterOnlyRGBVector.rows, 1, CV_32SC1);
		std::printf("WaterOnlyLabelVector_float dims: %d\t%d\t%d\t\n", WaterOnlyLabelVector_float.rows, WaterOnlyLabelVector_float.cols, WaterOnlyLabelVector_float.channels());
		//assign value to 
		if (!waterTrainingRGBVector.data) {
			//if this the first time assigning
			// no need to vconcat
			waterTrainingRGBVector = WaterOnlyRGBVector;
			std::printf("dims of waterTrainingRGBVector: %d\t%d\t%d\t\n", waterTrainingRGBVector.rows, waterTrainingRGBVector.cols, waterTrainingRGBVector.channels());
		}//end of if
		else {
			cv::vconcat(waterTrainingRGBVector, WaterOnlyRGBVector, waterTrainingRGBVector);
			std::printf("dims of waterTrainingRGBVector: %d\t%d\t%d\t\n", waterTrainingRGBVector.rows, waterTrainingRGBVector.cols, waterTrainingRGBVector.channels());
		}//end of else

		//////////////////////////////
		//labeling
		// for label +1, Hue water label
		if (i <= 6) {
			if (!waterTrainingLabelsVector_float.data) {
				waterTrainingLabelsVector_float = WaterOnlyLabelVector_float;
				std::printf("waterTrainingLabelsVector_float dims: %d\t%d\t%d\t\n\n", waterTrainingLabelsVector_float.rows, waterTrainingLabelsVector_float.cols, waterTrainingLabelsVector_float.channels());
			}
			else {
				cv::vconcat(waterTrainingLabelsVector_float, WaterOnlyLabelVector_float, waterTrainingLabelsVector_float);
				std::printf("waterTrainingLabelsVector_float dims: %d\t%d\t%d\t\n\n", waterTrainingLabelsVector_float.rows, waterTrainingLabelsVector_float.cols, waterTrainingLabelsVector_float.channels());
			} // end of else
		} // end if i<=6
		// label +2.0 for Saturation water
		else if (i > 6 && i <= 11) {
			WaterOnlyLabelVector_float += 1.0;
			cv::vconcat(waterTrainingLabelsVector_float, WaterOnlyLabelVector_float, waterTrainingLabelsVector_float);
			std::printf("waterTrainingLabelsVector_float dims: %d\t%d\t%d\t\n\n", waterTrainingLabelsVector_float.rows, waterTrainingLabelsVector_float.cols, waterTrainingLabelsVector_float.channels());
		}
		if (i == 12) {
			WaterOnlyLabelVector_float += 2.0;
			cv::vconcat(waterTrainingLabelsVector_float, WaterOnlyLabelVector_float, waterTrainingLabelsVector_float);
			std::printf("waterTrainingLabelsVector_float dims: %d\t%d\t%d\t\n\n", waterTrainingLabelsVector_float.rows, waterTrainingLabelsVector_float.cols, waterTrainingLabelsVector_float.channels());
		}//end if i==12
		if (i == 13) {
			WaterOnlyLabelVector_float += 3.0;
			cv::vconcat(waterTrainingLabelsVector_float, WaterOnlyLabelVector_float, waterTrainingLabelsVector_float);
			std::printf("waterTrainingLabelsVector_float dims: %d\t%d\t%d\t\n\n", waterTrainingLabelsVector_float.rows, waterTrainingLabelsVector_float.cols, waterTrainingLabelsVector_float.channels());
		}//end if i==13
		if (i == 14 || i == 15) {
			WaterOnlyLabelVector_float += 4.0;
			cv::vconcat(waterTrainingLabelsVector_float, WaterOnlyLabelVector_float, waterTrainingLabelsVector_float);
			std::printf("waterTrainingLabelsVector_float dims: %d\t%d\t%d\t\n\n", waterTrainingLabelsVector_float.rows, waterTrainingLabelsVector_float.cols, waterTrainingLabelsVector_float.channels());
		}//end if i==14 || i==15


	}//end of for reading water training samples

	// assign the output
	waterTrainingFeatureVectorRGB_uchar = waterTrainingRGBVector;
	waterTrainingLabelVector_float = waterTrainingLabelsVector_float;

	}// end of function createWaterOnlyHSVFeatureVector


	int CreateNonWaterHSVFeatureVector(Mat& NonWaterFetureVectorRGB_uchar, Mat& NonWaterLabelVector_float) {

	//read an image
	Mat image = imread("9554.png", 1);
	//check for existance of data
	if (!image.data)
	{
		std::printf("no image data.\n"); return -1;
	}

	// defining non water coordination
	Rect nonWaterRect(1, 400, 640, 320);
	//non water image
	Mat NonWaterImage = image(nonWaterRect);
	//imshow("non water image", NonWaterImage);

	// convert to hsv
	cv::cvtColor(NonWaterImage, NonWaterImage, cv::COLOR_BGR2HSV);

	//holding plited nonwater image planes
	Mat NonWaterPlanes[3];

	//split nonwater image sample
	split(NonWaterImage, NonWaterPlanes);
	std::printf("NonWaterPlanes[0] dims: %d\t%d\t%d \n", NonWaterPlanes[0].rows, NonWaterPlanes[0].cols, NonWaterPlanes[0].channels());

	// 3 column vector for each of rgb planes
	Mat NonWaterValueVector, NonWaterSaturationVector, NonWaterHueVector;

	// reshaping each plane to get a column vector Mx1
	NonWaterValueVector = NonWaterPlanes[2].reshape(0, 1).t();//Value 1st
	NonWaterSaturationVector = NonWaterPlanes[1].reshape(0, 1).t();//Saturation 2nd
	NonWaterHueVector = NonWaterPlanes[0].reshape(0, 1).t();//Hue
	std::printf("NonWaterSaturationVector size is: %d\t%d\t%d\t\n", NonWaterSaturationVector.rows, NonWaterSaturationVector.cols, NonWaterSaturationVector.channels());

	// Mx3 vector holding all RGB pixels of NonWater
	Mat NonWaterRGBVector;

	// concatenate 3 column vec into one place (Mx3)
	cv::hconcat(NonWaterValueVector, NonWaterSaturationVector, NonWaterRGBVector);
	cv::hconcat(NonWaterRGBVector, NonWaterHueVector, NonWaterRGBVector);
	std::printf("NonWaterRGBVector dims: %d\t%d\t%d\t\n", NonWaterRGBVector.rows, NonWaterRGBVector.cols, NonWaterRGBVector.channels());

	// label vector for NonWater pixels
	NonWaterLabelVector_float = cv::Mat::zeros(NonWaterRGBVector.rows, 1, CV_32SC1);
	std::printf("NonWaterLabelVector_float dims: %d\t%d\t%d\t\n", NonWaterLabelVector_float.rows, NonWaterLabelVector_float.cols, NonWaterLabelVector_float.channels());
	//cout << NonWaterLabelVector_float << "\n";
	//cout << NonWaterLabelVector_float << "\n";

	// assign output
	NonWaterFetureVectorRGB_uchar = NonWaterRGBVector;
	}// end of function createWaterOnlyHSVFeatureVector

	void printminMaxLoc(Mat& in) {
		double min, max;
		cv::minMaxLoc(in, &min, &max);
		std::cout << "min and max:\t" << min << "\t" << max << "\n";
	}// end printminmaxloc

