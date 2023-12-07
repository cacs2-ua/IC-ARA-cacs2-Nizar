#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <future>
#include "libraw/libraw.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <algorithm>
#include <future>
#include <omp.h>
#include <vector>
#define SQR(x) ((x) * (x))

using namespace std;
using namespace std::chrono;

void colorBalance(cv::Mat& in, cv::Mat& out, float percent);
void gammaCorrection(cv::Mat& in, cv::Mat& out, float a, float b, float gamma);
void sharpening(cv::Mat& in, cv::Mat& out, float sigma, float amount);
void enhanceDetails(cv::Mat &in, cv::Mat &out, float sigma, float amoount);
void bloom(cv::Mat &in, cv::Mat &out, float sigma, float threshold);
void denoise(cv::Mat &in, cv::Mat &out, float sigma);
void equalization(cv::Mat &in, cv::Mat &out, float black, float white, float saturation);
void debayer(LibRaw* processor, cv::Mat &out);
void screenMerge(cv::Mat &in1, cv::Mat &in2, cv::Mat &out);

void gammaCurve(unsigned short *curve, double power)
{
    auto start = high_resolution_clock::now();
    double pwr = 1.0 / power;
    double ts = 0.0;
    int imax = 0xffff;
    int mode = 2;
    int i;
    double g[6], bnd[2] = {0, 0}, r;

    g[0] = pwr;
    g[1] = ts;
    g[2] = g[3] = g[4] = 0;
    bnd[g[1] >= 1] = 1;
    if (g[1] && (g[1] - 1) * (g[0] - 1) <= 0)
    {
        for (i = 0; i < 48; i++)
        {
            g[2] = (bnd[0] + bnd[1]) / 2;
            if (g[0])
                bnd[(pow(g[2] / g[1], -g[0]) - 1) / g[0] - 1 / g[2] > -1] = g[2];
            else
                bnd[g[2] / exp(1 - 1 / g[2]) < g[1]] = g[2];
        }
        g[3] = g[2] / g[1];
        if (g[0])
            g[4] = g[2] * (1 / g[0] - 1);
    }
    if (g[0])
        g[5] = 1 / (g[1] * SQR(g[3]) / 2 - g[4] * (1 - g[3]) +
            (1 - pow(g[3], 1 + g[0])) * (1 + g[4]) / (1 + g[0])) -
            1;
    else
        g[5] = 1 / (g[1] * SQR(g[3]) / 2 + 1 - g[2] - g[3] -
            g[2] * g[3] * (log(g[3]) - 1)) -
           1;
    #pragma omp parallel for private(r)
    for (i = 0; i < 0x10000; i++)
    {
        curve[i] = 0xffff;
        if ((r = (double)i / imax) < 1)
        {
          curve[i] =
              0x10000 *
              (mode ? (r < g[3] ? r * g[1]
                                : (g[0] ? pow(r, g[0]) * (1 + g[4]) - g[4]
                                        : log(r) * g[2] + 1))
                    : (r < g[2] ? r / g[1]
                                : (g[0] ? pow((r + g[4]) / (1 + g[4]), 1 / g[0])
                                        : exp((r - 1) / g[2]))));
        }
    }
    auto end = high_resolution_clock::now();
    auto elapsed_ms = duration_cast<milliseconds>(end - start);

    cout<<"Gamma curve creation: "<<elapsed_ms.count()<<"ms"<<endl;
}

void equalization(cv::Mat &in, cv::Mat &out, float black, float white, float saturation)
{
	auto start = high_resolution_clock::now();
	cv::Mat hsv;
	// convert to float32
	in.convertTo(hsv, CV_32F, 1.0/65535.0);
	// convert to HSV color space
	cv::cvtColor(hsv, hsv, cv::COLOR_BGR2HSV);
	
	// split HSV into 3 channels
	std::vector<cv::Mat> channels;
	cv::split(hsv, channels);
	// normalize values between minimum black level and maximum white level
	cv::normalize(channels[2], channels[2], black, white, cv::NORM_MINMAX);
	// increase saturation
	channels[1] *= (1 + saturation);
	// convert back to RGB and 16 bit
	cv::merge(channels, out);
	cv::cvtColor(out, out, cv::COLOR_HSV2BGR);
	out.convertTo(out, CV_16U, 65535);
	
	auto end = high_resolution_clock::now();
	auto elapsed_ms = duration_cast<milliseconds>(end - start);

	cout<<"Equalization: "<<elapsed_ms.count()<<"ms"<<endl;
}

void denoise(cv::Mat &in, cv::Mat &out, int windowSize)
{
	auto start = high_resolution_clock::now();
	
	cv::Mat ycrcb;
	// convert to float32
	in.convertTo(ycrcb, CV_32F, 1.0/65535.0);
	// convert to YCrCb color space
	cv::cvtColor(ycrcb, ycrcb, cv::COLOR_BGR2YCrCb);
	
	// split ycrcb into 3 channels
	std::vector<cv::Mat> channels;
	cv::split(ycrcb, channels);

    // remove noise from chrominance channels
	cv::medianBlur(channels[1], channels[1], windowSize);
	cv::medianBlur(channels[2], channels[2], windowSize);

    // convert back to RGB and 16 bits
	cv::merge(channels, out);
	cv::cvtColor(out, out, cv::COLOR_YCrCb2BGR);
	out.convertTo(out, CV_16U, 65535);
	
	auto end = high_resolution_clock::now();
	auto elapsed_ms = duration_cast<milliseconds>(end - start);

	cout<<"Denoise: "<<elapsed_ms.count()<<"ms"<<endl;
}

void debayer(LibRaw* processor, cv::Mat &out)
{
    auto start = high_resolution_clock::now();
    processor->raw2image();
    int width = processor->imgdata.sizes.iwidth;
    int height = processor->imgdata.sizes.iheight;
    int orientation = processor->imgdata.sizes.flip;
    
    // create a buffer of ushorts containing the single channel bayer pattern
    //std::vector<ushort> bayerData;
    cv::Mat imgBayer(height, width, CV_16UC1);
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            // Get pixel index
            int idx = y * width + x;

            // Each pixel is an array of 4 shorts rgbg
            ushort *rgbg = processor->imgdata.image[idx];

            // Determine the pixel value based on Bayer pattern
            ushort pixelValue;
            if (y % 2 == 0) // Even rows
            {
                pixelValue = rgbg[x % 2 == 0 ? 0 : 1]; // Red if x is even, green if odd
            }
            else // Odd rows
            {
                pixelValue = rgbg[x % 2 == 0 ? 3 : 2]; // Green if x is even, blue if odd
            }

            imgBayer.at<ushort>(y, x) = pixelValue;
        }
    }

    // create an OpenCV matrix with the bayer pattern
    //cv::Mat imgBayer(height, width, CV_16UC1, bayerData.data());
    //cv::Mat imgDeBayer;
    // apply the debayering algorithm
    cv::cvtColor(imgBayer, out, cv::COLOR_BayerBG2BGR);
    //out = imgDeBayer;
    
    switch(orientation)
    {
        case 2:
            cv::flip(out, out, 0);
            break;
        case 3:
            cv::rotate(out, out, cv::ROTATE_180);
            break;
        case 4:
            cv::flip(out, out, 1);
            break;
        case 5:
            cv::rotate(out, out, cv::ROTATE_90_COUNTERCLOCKWISE);
            break;
        case 6:
            cv::rotate(out, out, cv::ROTATE_90_CLOCKWISE);
            break;
        case 7:
            cv::flip(out, out, 0);
            cv::rotate(out, out, cv::ROTATE_90_CLOCKWISE);
            break;
        case 8:
            cv::flip(out, out, 0);
            cv::rotate(out, out, cv::ROTATE_90_COUNTERCLOCKWISE);
            break;
    }
    
    auto end = high_resolution_clock::now();
	auto elapsed_ms = duration_cast<milliseconds>(end - start);

	cout<<"De Bayer: "<<elapsed_ms.count()<<"ms"<<endl;
}

void sharpening(cv::Mat& in, cv::Mat& out, float sigma, float amount)
{
    auto start = high_resolution_clock::now();

    cv::Mat blurry;
    // Create a blurred image
    cv::GaussianBlur(in, blurry, cv::Size(), sigma);

    // Ensure out has the same size and type as in
    out.create(in.size(), in.type());

    // Parallelize this loop with OpenMP
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < in.rows; ++y)
    {
        for (int x = 0; x < in.cols; ++x)
        {
            for (int c = 0; c < in.channels(); ++c)
            {
                out.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(
                    in.at<cv::Vec3b>(y, x)[c] * (1 + amount) - blurry.at<cv::Vec3b>(y, x)[c] * amount
                );
            }
        }
    }

    auto end = high_resolution_clock::now();
    auto elapsed_ms = duration_cast<milliseconds>(end - start);

    cout << "Sharpening: " << elapsed_ms.count() << "ms" << endl;
}




void processSliceEnhanced(cv::Mat& inFloat, cv::Mat& blur, int startRow, int endRow, int cols, float amount) {
    for (int i = startRow; i < endRow; ++i) {
        float* pIn = inFloat.ptr<float>(i);
        float* pBlur = blur.ptr<float>(i);
        for (int j = 0; j < cols; ++j) {
            for (int c = 0; c < 3; c++) {
                float im = pIn[j * 3 + c];
                float b = pBlur[j * 3 + c];
                float d = im - b;
                pBlur[j * 3 + c] = b + d * amount;
            }
        }
    }
}

void enhanceDetails(cv::Mat &in, cv::Mat &out, float sigma, float amount)
{
    auto start = high_resolution_clock::now();

    cv::Mat blur, inFloat;
    // Convert to float32
    in.convertTo(inFloat, CV_32F, 1.0 / 65535);
    // Create a blurred image
    cv::GaussianBlur(inFloat, blur, cv::Size(), sigma);

    int numThreads = std::thread::hardware_concurrency();
    int sliceHeight = in.rows / numThreads;
    std::vector<std::future<void>> futures;

    for (int t = 0; t < numThreads; ++t) {
        int startRow = t * sliceHeight;
        int endRow = (t == numThreads - 1) ? in.rows : startRow + sliceHeight;
        futures.push_back(std::async(std::launch::async, processSliceEnhanced, std::ref(inFloat), std::ref(blur), startRow, endRow, in.cols, amount));
    }

    // Wait for all tasks to complete
    for (auto& f : futures) {
        f.get();
    }

    // Convert back to 16 bit
    blur.convertTo(out, CV_16U, 65535);

    auto end = high_resolution_clock::now();
    auto elapsed_ms = duration_cast<milliseconds>(end - start).count();

    cout << "Enhanced details: " << elapsed_ms << "ms" << endl;
}



void bloom(cv::Mat &in, cv::Mat &out, float sigma, float threshold)
{
	auto start = high_resolution_clock::now();
	
    cv::Mat blur, mask, inFloat;
    // convert to float32
    in.convertTo(inFloat, CV_32F, 1.0/65536);
    cv::Mat ycrcb;
    // convert to YCrCb color space
	cv::cvtColor(inFloat, ycrcb, cv::COLOR_BGR2YCrCb);
	
	// split YCrCb into 3 channels
	cv::Mat channels[3];
	cv::split(ycrcb, channels);
    
    // normalize Y channel between 0 and 1
    cv::normalize(channels[0], mask, 0.0, 1.0, cv::NORM_MINMAX);
    // set to 1.0 only pixels above the threshold
    cv::threshold(mask, mask, threshold, 1.0, cv::THRESH_BINARY);
    // apply gaussian blur to thresholded pixels
    cv::GaussianBlur(mask, mask, cv::Size(), sigma);
    
    // convert the computed mask to 3 channel image and 16 bit
    cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
    mask.convertTo(out, CV_16U, 65535);
    
    auto end = high_resolution_clock::now();
	auto elapsed_ms = duration_cast<milliseconds>(end - start);

	cout << "Bloom: " << elapsed_ms.count()<<"ms"<<endl;
}

void gammaCorrection(cv::Mat& in, cv::Mat& out, float a, float b, float gamma)
{
    auto start = high_resolution_clock::now();

    cv::Mat tmp = cv::Mat::zeros(in.size(), in.type());
    unsigned short curve[0x10000];
    // Create the gamma LUT
    gammaCurve(curve, gamma);

    // Parallelize this loop with OpenMP
    #pragma omp parallel for
    for (int i = 0; i < in.rows; ++i)
    {
        unsigned short* p = in.ptr<unsigned short>(i);
        unsigned short* tp = tmp.ptr<unsigned short>(i);
        for (int j = 0; j < in.cols; ++j)
        {
            tp[j * 3] = a * curve[p[j * 3]] + b;
            tp[j * 3 + 1] = a * curve[p[j * 3 + 1]] + b;
            tp[j * 3 + 2] = a * curve[p[j * 3 + 2]] + b;
        }
    }
    out = tmp;

    auto end = high_resolution_clock::now();
    auto elapsed_ms = duration_cast<milliseconds>(end - start);

    cout << "Gamma correction: " << elapsed_ms.count() << "ms" << endl;
}

void colorBalance(cv::Mat& in, cv::Mat& out, float percent) {

    auto start = high_resolution_clock::now();

    float half_percent = percent / 200.0f;

    std::vector<cv::Mat> tmpsplit; 
    cv::split(in, tmpsplit);
    int max = (in.depth() == CV_8U ? 1 << 8 : 1 << 16) - 1;

    // Parallelize this loop with OpenMP
    #pragma omp parallel for
    for (int i = 0; i < 3; i++) 
    {
        // Find the low and high percentile values (based on the input percentile)
        cv::Mat flat;
        tmpsplit[i].reshape(1, 1).copyTo(flat);
        cv::sort(flat, flat, cv::SORT_EVERY_ROW | cv::SORT_ASCENDING);
        int lowval = flat.at<ushort>(cvFloor(static_cast<float>(flat.cols) * half_percent));
        int highval = flat.at<ushort>(cvCeil(static_cast<float>(flat.cols) * (1.0 - half_percent)));

        // Saturate below the low percentile and above the high percentile
        tmpsplit[i].setTo(lowval, tmpsplit[i] < lowval);
        tmpsplit[i].setTo(highval, tmpsplit[i] > highval);

        // Scale the channel
        cv::normalize(tmpsplit[i], tmpsplit[i], 0, max, cv::NORM_MINMAX);
    }
    cv::merge(tmpsplit, out);

    auto end = high_resolution_clock::now();
    auto elapsed_ms = duration_cast<milliseconds>(end - start);

    cout << "Color balance: " << elapsed_ms.count() << "ms" << endl;
}

void screenMerge(cv::Mat &in1, cv::Mat &in2, cv::Mat &out)
{
    auto start = high_resolution_clock::now();
    
    cv::Mat inFloat1, inFloat2;
    // Convert to float32 to avoid overflow
    in1.convertTo(inFloat1, CV_32F, 1.0/65535);
    in2.convertTo(inFloat2, CV_32F, 1.0/65535);
    cv::Mat tmp = cv::Mat::zeros(inFloat1.size(), inFloat1.type());

    // Parallelize this loop with OpenMP
    #pragma omp parallel for
    for (int i = 0; i < in1.rows; ++i)
    {
        float* pIn1 = inFloat1.ptr<float>(i);
        float* pIn2 = inFloat2.ptr<float>(i);
        float* pTmp = tmp.ptr<float>(i);
        for (int j = 0; j < in1.cols; ++j)
        {
            for (int c = 0; c < 3; c++)
            {
                float im = pIn1[j * 3 + c];
                float m = pIn2[j * 3 + c];
                pTmp[j * 3 + c] = 1.0 - (1.0 - m) * (1.0 - im);
            }
        }
    }
    tmp.convertTo(out, CV_16U, 65535);
    
    auto end = high_resolution_clock::now();
    auto elapsed_ms = duration_cast<milliseconds>(end - start);

    cout << "Screen mode merge: " << elapsed_ms.count() << "ms" << endl;
}

int main(int argc, char *argv[])
{
    auto start = high_resolution_clock::now();

    int ret;
    if (argc < 3)
    {
        cerr<<"Raw file needed as first argument. Usage 'pipeline <input_file_name> <output_file_name>'."<<endl;
        
        return 0;
    }
    
    string inputFile = argv[1];
    string outputFile = argv[2];
    LibRaw* processor = new LibRaw;
    
    // load the raw image
    if ((ret = processor->open_file(inputFile.c_str())) != LIBRAW_SUCCESS)
    {
        cerr<<"Cannot open file "<<inputFile<<": "<<libraw_strerror(ret)<<endl;
        
        return 0;
    }
    
    // unpack the values
    if (ret = processor->unpack() != LIBRAW_SUCCESS)
    {
        cerr<<"Cannot do postprocessing on "<<inputFile<<": "<<libraw_strerror(ret)<<endl;
        
        return 0;
    }
    
    cv::Mat image;
    // debayer the raw image
    debayer(processor, image);
    delete processor;
    
    // remove chrominance noise
    denoise(image, image, 5);
    // apply gamma correction (move from linear output to non linear)
    gammaCorrection(image, image, 1.0, 0.0, 2.2);
    // apply color balance correction
    colorBalance(image, image, 2);
    // equalize luminance values and increase saturation
    equalization(image, image, 0.0, 1.0, 0.5);
    cv::Mat enhanced, bloomed;
    // enhance high frequency details
    //enhanceDetails(image, enhanced, 20, 1.25);
    // compute bloom mask
    //bloom(image, bloomed, 70, 0.9);
    auto enhanceFuture = std::async(std::launch::async, [&]() {
        enhanceDetails(image, enhanced, 20, 1.25);
    });
    auto bloomFuture = std::async(std::launch::async, [&]() {
        bloom(image, bloomed, 70, 0.9);
    });

    // Wait for the results
    enhanceFuture.get();
    bloomFuture.get();
    // combine enhanced details with bloom mask
    screenMerge(enhanced, bloomed, image);
    // convert to 8 bit image
    image.convertTo(image, CV_8U, 1.0/255.0);
    // save final image
    cv::imwrite(outputFile, image);
    

    auto end = high_resolution_clock::now();

    // Calculate the total execution time
    auto duration = duration_cast<milliseconds>(end - start).count();
    cout << "Total execution time: " << duration << "ms" << endl;

    return 0;
}
