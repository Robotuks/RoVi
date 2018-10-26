#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

using namespace cv;
using namespace std;

int KERNEL_LENGTH=31;
int DELAY_CAPTION = 1500;
int DELAY_BLUR = 100;

Mat histogram(const Mat& img)
{
    assert(img.type() == CV_8UC1);

    Mat hist;
    calcHist(
            vector<Mat>{img},
            {0}, // channels the value is zero because we have grayscale image
            noArray(), // mask
            hist, // output histogram
            {256}, // histogram sizes (number of bins) in each dimension
            {0, 256} // pairs of bin lower (incl.) and upper (excl.) boundaries in each dimension
        );
        return hist; // returned type is 32FC1
}

Mat draw_histogram_image(const Mat& hist)
{
    int nbins = hist.rows;
    double max = 0;
    minMaxLoc(hist, nullptr, &max);
   Mat img(nbins, nbins, CV_8UC1,Scalar(255));

    for (int i = 0; i < nbins; i++) {
        double h = nbins * (hist.at<float>(i) / max); // Normalize
        line(img, Point(i, nbins), Point(i, nbins - h), Scalar::all(0));
    }

    return img;
}

// Rearranges the quadrants of a Fourier image so that the origin is at the
// center of the image.
void dftshift(Mat& mag)
{
    int cx = mag.cols / 2;
    int cy = mag.rows / 2;

    Mat tmp;
    Mat q0(mag, Rect(0, 0, cx, cy));
    Mat q1(mag, Rect(cx, 0, cx, cy));
    Mat q2(mag, Rect(0, cy, cx, cy));
    Mat q3(mag, Rect(cx, cy, cx, cy));

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

Mat pad(Mat& src)
{
    Mat padded;
    int opt_rows = getOptimalDFTSize(src.rows * 2 - 1);
    int opt_cols = getOptimalDFTSize(src.cols * 2 - 1);
    copyMakeBorder(src,
        padded,
        0,
        opt_rows - src.rows,
        0,
        opt_cols - src.cols,
        BORDER_CONSTANT,
        Scalar::all(0)
    );
    return padded;
}

void frequency_magnitude(Mat planes[2])
{

    Mat complex;
    merge(planes, 2, complex);

    // Compute DFT
    dft(complex, complex);
    // Split real and complex planes
    split(complex, planes);
}


Mat magnitude_to_img(Mat& magnitude, Mat& phase, Mat planes[2], Mat& img)
{
    // Shift back quadrants of the spectrum
    dftshift(magnitude);

    // Compute complex DFT planes from magnitude/phase
    polarToCart(magnitude, phase, planes[0], planes[1]);
    Mat filtered, complex;
    // Merge into one image
    merge(planes, 2, complex);
    idft(complex, filtered, (cv::DFT_SCALE | cv::DFT_REAL_OUTPUT));

    // Crop image (remove padded borders)
    filtered = cv::Mat(filtered, cv::Rect(cv::Point(0, 0), img.size()));
    dftshift(magnitude);
    return filtered;
}

void show_frequency_magnitude(Mat& mag, string name)
{
    Mat mag1 = mag;
    mag1 += Scalar::all(1);
    log(mag1, mag1);
    normalize(mag1, mag1, 0, 1, NORM_MINMAX);
    namedWindow( name, WINDOW_NORMAL  );
    imshow(name, mag1);
}

Mat contraharmonicFilter(Mat input_image, Mat filter)
{
    Mat output_image(input_image.rows, input_image.cols, input_image.type());
    input_image.copyTo(output_image);
    double sum[3], sum1[3];
    double Q = filter.ptr<double>(0)[0];

    for (int r = 0; r < output_image.rows; r++) {
        for (int c = 0; c < output_image.cols; c++) {
            sum[0] = sum[1] = sum[2] = 0;
            sum1[0] = sum1[1] = sum1[2] = 0;

            int i = (r-filter.rows/2<0)?0:r-filter.rows/2;
            for (; i < output_image.rows && i <= r+filter.rows/2; i++) {
                int j = (c - filter.cols/2<0)?0:c-filter.cols/2;
                for (; j < output_image.cols && j <= c+filter.cols/2; j++) {
                    for (int n = 0; n < output_image.channels()&&n<3; n++) {
                        if (output_image.channels()==1) {
                            sum[n] += pow(input_image.ptr<uchar>(i)[j], 1+Q);
                            sum1[n] += pow(input_image.ptr<uchar>(i)[j], Q);
                        } else if (output_image.channels() == 3) {
                            sum[n] += pow(input_image.at<Vec3b>(i, j)[n], 1+Q);
                            sum1[n] += pow(input_image.at<Vec3b>(i, j)[n], Q);
                        } else {
                            sum[n] += pow(input_image.at<Vec4b>(i, j)[n], 1+Q);
                            sum1[n] += pow(input_image.at<Vec4b>(i, j)[n], Q);
                        }
                    }
                }
            }
             for (int n = 0; n < output_image.channels() && n<3; n++) {
                if (output_image.channels()==1)        output_image.ptr<uchar>(r)[c] = saturate_cast<uchar>(sum[n]/sum1[n]);
                else if (output_image.channels() == 3) output_image.at<Vec3b>(r, c)[n] = saturate_cast<uchar>(sum[n]/sum1[n]);
                else output_image.at<Vec4b>(r, c)[n] = saturate_cast<uchar>(sum[n]/sum1[n]);
            }
        }
    }
    return output_image;
}

int main(int argc, char* argv[])
{
    // Parse command line arguments -- the first positional argument expects an
    // image path (the default is ./book_cover.jpg)
    CommandLineParser parser(argc, argv,
        // name  | default value    | help message
        "{help   |                  | print this message}"
        "{@image | /home/student/Desktop/RoVi/Images/ImagesForStudents/Image4_2.png | image path}"
        "{@img_number | 41 | }"
    );

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    // Load image file
    string filepath = parser.get<string>("@image");
    Mat img =imread(filepath, IMREAD_GRAYSCALE);

    // Check that the image file was actually loaded
    if (img.empty()) {
        cout << "Input image not found at '" << filepath << "'\n";
        return 1;
    }
  /* Mat padded;
            int opt_rows = getOptimalDFTSize(img.rows*2-1);
            int opt_cols = getOptimalDFTSize(img.cols*2-1);
            copyMakeBorder(img,padded,0,opt_rows-img.rows,0,opt_cols-img.cols,BORDER_CONSTANT,Scalar::all(0));

            //namedWindow("OriginalImage",WINDOW_NORMAL);
            //namedWindow("Padded",WINDOW_NORMAL);
          //  imshow("OriginalImage",img);
            //imshow("Padded",padded);
*/

    if (parser.get<int>("@img_number") == 1 )
    {
    
    Mat ft = (Mat_<double>(7,7)<<0.5); //7*7 is the kernel lenth, sec arg>0 remove pepper noise, <0 salt noise
    Mat dst=contraharmonicFilter(img, ft);

    }
    else if (parser.get<int>("@img_number") == 3 )
    {
        namedWindow("ROI",WINDOW_NORMAL);
        Rect2d roi = selectROI("ROI",img);
        // Crop image
        Mat Crop = img(roi);
        Mat hist_crp=histogram(Crop);
        imshow("Cropped Histogram",draw_histogram_image(hist_crp));
        Mat tmp;
        Mat fast;
        /*for ( int i = 3; i < 12; i = i + 2 )
        {
            GaussianBlur( img, tmp, Size( i, i ), 0, 0 );
            //medianBlur ( tmp, median, i );
            namedWindow(to_string(i),WINDOW_NORMAL);
            imshow(to_string(i), tmp);
        }*/
        fastNlMeansDenoising(img,fast, 20, 5, 35);
        namedWindow("fast",WINDOW_NORMAL);
        imshow("fast", fast);
        //GaussianBlur( img_padded, tmp, Size( KERNEL_LENGTH, KERNEL_LENGTH ), 0, 0 );
        namedWindow("OriginalImage",WINDOW_NORMAL);
        imshow("OriginalImage", img);
    }
    else if (parser.get<int>("@img_number") == 41 )
    {
        Mat img_padded = pad(img);
        Mat planes[] = {
                Mat_<float>(img_padded),
                Mat_<float>::zeros(img_padded.size())
        };
        frequency_magnitude( planes);
        // Compute the magnitude and phase
        Mat magnitude, phase;
        cartToPolar(planes[0], planes[1], magnitude, phase);

        // Shift quadrants so the Fourier image origin is in the center of the image
        dftshift(magnitude);

        show_frequency_magnitude(magnitude, "Magnitude");

        namedWindow("ROI2",WINDOW_NORMAL);
        Rect2d roi2 = selectROI("ROI2",magnitude);
        rectangle(magnitude, roi2, Scalar(0), CV_FILLED);
        namedWindow("ROI3",WINDOW_NORMAL);
        Rect2d roi3 = selectROI("ROI3",magnitude);
        rectangle(magnitude, roi3, Scalar(0), CV_FILLED);

        show_frequency_magnitude(magnitude, "Magnitude_cropped");
        Mat filtered = magnitude_to_img(magnitude, phase, planes, img);
        namedWindow("FilteredImage",WINDOW_NORMAL);
        //cv::normalize(filtered, filtered, 0, 1, cv::NORM_MINMAX);
        imshow("FilteredImage", filtered);
        namedWindow("OriginalImage",WINDOW_NORMAL);
        imshow("OriginalImage", img);
    }
    // Display Cropped Image
    // namedWindow("Imagecr",WINDOW_AUTOSIZE);
    //  imshow("Imagecr", imCrop);
     // Show the image  */



    //namedWindow("Median FIlter",WINDOW_NORMAL);
    //imshow("Median FIlter", median);



    // Wait for escape key press before returning
    while (waitKey() != 27)
        ; // (do nothing)

    return 0;
}
