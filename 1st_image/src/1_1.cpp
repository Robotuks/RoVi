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

Mat frequency_spectrum(Mat& src)
{

    Mat planes[] = {
                   Mat_<float>(src),
                   Mat_<float>::zeros(src.size())
    };

    Mat_<cv::Vec2f>complex;
    merge(planes, 2, complex);

    // Compute DFT
    dft(complex, complex);
    // Split real and complex planes
   // split(complex, planes);

    return complex;
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

void show_frequency_spectrum(Mat& mag, string name)
{
    Mat mag1 ;
    mag.copyTo(mag1);
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

Mat butter_lowpass(float d0, int n, Size size)
{
    cv::Mat_<cv::Vec2f> lpf(size);
    cv::Point2f c = cv::Point2f(size) / 2;

    for (int i = 20; i < size.height; ++i) {
        for (int j = 20; j < size.width; ++j) {
            // Distance from point (i,j) to the origin of the Fourier transform
            float d = std::sqrt((i - c.y) * (i - c.y) + (j - c.x) * (j - c.x));

            // Real part
            lpf(i, j)[0] = 1 / (1 + std::pow(d / d0, 2 * n));

            // Imaginary part
            lpf(i, j)[1] = 0;
        }
    }

    return lpf;
}

Mat butter_highpass(float d0, int n, cv::Size size)
{
    cv::Mat_<cv::Vec2f> hpf(size);
    cv::Point2f c = cv::Point2f(size) / 2;

    for (int i = 0; i < size.height; ++i) {
        for (int j = 0; j < size.width; ++j) {
            // Distance from point (i,j) to the origin of the Fourier transform
            float d = std::sqrt((i - c.y) * (i - c.y) + (j - c.x) * (j - c.x));

            // Real part
            if (std::abs(d) < 1.e-9f) // Avoid division by zero
                hpf(i, j)[0] = 0;
            else {
                hpf(i, j)[0] = 1 / (1 + std::pow(d0 / d, 2 * n));
            }

            // Imaginary part
            hpf(i, j)[1] = 0;
        }
    }

    return hpf;
}
float getRadius(int cols, int rows, float rad){
    int min = cols;
    if(rows < cols){
        min = rows;
    }
    return rad*min/2;
}
void bandRejectFilter(Mat &imgReal, float low, float high){
    int rows = imgReal.rows;
    int cols = imgReal.cols;
    float h = getRadius(cols, rows, high);
    float l = getRadius(cols, rows, low);
    float high2 = h*h;
    float low2 = l*l;
    for(int i = 0; i < cols; i++){
        for(int j = 0; j < rows; j++){
            int dj = (j < rows / 2) ? j : rows - j;
            int di = (i < cols / 2) ? i : cols - i;
            float dist2 = dj*dj + di*di;
            if(dist2 > low2 && dist2 < high2){
                imgReal.at<float>(j,i) = 0;
                           }
        }
    }
}

Mat apply_filter(Mat cmp, Mat filter, Mat img)
{
     mulSpectrums(cmp, filter, cmp, 0);
     // Multiply Fourier image with filter
     mulSpectrums(cmp, filter, cmp, 0);

     // Shift back
     dftshift(cmp);
     // Compute inverse DFT
     Mat filtered;
     idft(cmp, filtered, (cv::DFT_SCALE | cv::DFT_REAL_OUTPUT));

     // Crop image (remove padded borders)
     filtered = Mat(filtered, cv::Rect(cv::Point(0, 0), img.size()));

     Mat filter_planes[2];
     split(filter, filter_planes); // We can only display the real part
     normalize(filter_planes[0], filter_planes[0], 0, 1, cv::NORM_MINMAX);
     namedWindow("Filter",WINDOW_NORMAL);
     imshow("Filter", filter_planes[0]);

     normalize(filtered, filtered, 0, 1, cv::NORM_MINMAX);
     return filtered;
}

void median(Mat& src, Mat& dst, int kSize){

    Mat _src;
    // creates a border around the source image to not have to verify conditions near the edge of the original image
    copyMakeBorder(src, _src, (kSize/2), (kSize/2), (kSize/2), (kSize/2), BORDER_REPLICATE);

    for(int i = (kSize/2); i < _src.rows-(kSize/2); i++){
        for(int j = (kSize/2); j < _src.cols-(kSize/2); j++){

            vector<uchar> pixels;

            for(int k = i-(kSize/2); k <= i+(kSize/2); k++){
                for(int w = j-(kSize/2); w <= j+(kSize/2); w++){
                    pixels.push_back(_src.at<uchar>(k,w));
                }
            }

            // sort the pixels of the window filter
            sort(pixels.begin(), pixels.end());

            // takes the middle
            dst.at<uchar>(i-(kSize/2),j-(kSize/2)) = pixels.at((kSize*kSize)/2);
        }
}
}
void adaptiveMedianFilter(Mat& src)
{

    int maskSize =3;
    int maxMaskSize=11;
    Mat img =src.clone(); //Image we're filtering
    Mat dest = src.clone();
    int xMax = img.rows-(maskSize-1)/2;     //These integers asures that we dont acces the image outside of it's borders
    int yMax = img.cols-(maskSize-1)/2;    //(masksize-1)-2 is due to the 1 pixel in the center we are looking at, and divide by two as only the mask part facing the border is the problem
    vector<int> medianVec;
    int median;
    int medianIndex = (maskSize*maskSize-1)/2; //The minus one is due to the index at vectors starting at 0
    for (int x = (maskSize-1)/2; x <= xMax; x++)
        {
        //These 2 for loops go through all of the picture
        for (int y = (maskSize-1)/2; y<= yMax; y++)
        {
            for(int maskX = -(maskSize-1)/2; maskX <= (maskSize-1)/2; maskX++) //Looping through the mask
            {
                for(int maskY = -(maskSize-1)/2; maskY <=(maskSize-1)/2; maskY++)
                {
                    medianVec.push_back(img.at<uchar>(x+maskX, y+maskY));
                }
            }
            sort(medianVec.begin(), medianVec.begin()+maskSize*maskSize); //Sorting the values in the mask and finding the median
            median = medianVec[medianIndex];
          // dest.at<uchar>(x, y) = median;

            if (median == 0 || median == 255) //If we still have salt or pepper in the median, we need to increase the mask size
            {
            medianVec.clear();
                while (maskSize <= maxMaskSize) //Keep increasing as long as our mask is under the max size
                {
                    maskSize = maskSize +2;
                    xMax = img.rows-(maskSize-1)/2; //We also need to update our indexes and bounds to not get any errors
                    yMax = img.cols-(maskSize-1)/2;
                    medianIndex = (maskSize*maskSize-1)/2; //The minus one is due to the index at vectors starting at 0
                    for(int maskX = -(maskSize-1)/2; maskX <= (maskSize-1)/2; maskX++) //Looping through the mask
                        for(int maskY = -(maskSize-1)/2; maskY <= (maskSize-1)/2; maskY++)
                        {
                            medianVec.push_back(img.at<uchar>(x+maskX, y+maskY));
                        }
                    sort(medianVec.begin(), medianVec.begin()+maskSize*maskSize); //Sorting the values in the mask and finding the median
                    median = medianVec[medianIndex];
                    medianVec.clear();

                    if(median != 0)
                        if(median != 255)
                            break;

                }
            }
             dest.at<uchar>(x, y) = median;
            //Setting our mask size and indexes back to normal
            maskSize = 3;
            xMax = img.rows-(maskSize-1)/2; //We also need to update our indexes and bounds to not get any errors
            yMax = img.cols-(maskSize-1)/2;
            medianIndex = (maskSize*maskSize-1)/2; //The minus one is due to the index at vectors starting at 0
            medianVec.clear();

            }
        }



    namedWindow("test",WINDOW_NORMAL);
    imshow("test",dest);

}

int main(int argc, char* argv[])
{
    // Parse command line arguments -- the first positional argument expects an
    // image path (the default is ./book_cover.jpg)
    CommandLineParser parser(argc, argv,
        // name  | default value    | help message
        "{help   |                  | print this message}"
        "{@image | /home/student/Desktop/RoVi/Images/ImagesForStudents/Image2.png | image path}"
        "{@img_number | 2 | }"
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


    if (parser.get<int>("@img_number") == 1 )
    {
    
    Mat ft = (Mat_<double>(7,7)<<0.5); //7*7 is the kernel lenth, sec arg>0 remove pepper noise, <0 salt noise
    Mat dst=contraharmonicFilter(img, ft);

    }

    if (parser.get<int>("@img_number") == 2 )
    {

        Mat tmp;
       Mat median;
       for ( int i = 3; i < 12; i = i + 2 )
       {
            GaussianBlur( img, tmp, Size( i, i ), 0, 0 );
            medianBlur ( tmp, median, i );
            //namedWindow(to_string(i),WINDOW_NORMAL);
           //imshow(to_string(i), tmp);
       }

       namedWindow("median",WINDOW_NORMAL);
       imshow("median",median);






    while (waitKey() != 27);

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

           Mat cmp=frequency_spectrum(img);
            split(cmp,planes);
           // Compute the magnitude and phase
           Mat magnitude, phase;
           cartToPolar(planes[0], planes[1], magnitude, phase);

           // Shift quadrants so the Fourier image origin is in the center of the image
           dftshift(magnitude);

           Mat mag_copy ;
           magnitude.copyTo(mag_copy);
           mag_copy += Scalar::all(1);
           log(mag_copy, mag_copy);
           normalize(mag_copy, mag_copy, 0, 1, NORM_MINMAX);

           vector<Rect> ROIs;
           namedWindow("ROI", WINDOW_NORMAL);
           selectROIs("ROI", mag_copy, ROIs);

           if(ROIs.size()<1)
                return 0;
           for (size_t i = 0; i < ROIs.size(); i++)
           {
               rectangle(magnitude, ROIs[i], Scalar(0), CV_FILLED);
           }

           show_frequency_spectrum(magnitude, "Magnitude_cropped");
           Mat filtered = magnitude_to_img(magnitude, phase, planes, img);
           namedWindow("FilteredImage",WINDOW_NORMAL);
           cv::normalize(filtered, filtered, 0, 1, cv::NORM_MINMAX);
           imshow("FilteredImage", filtered);
           namedWindow("OriginalImage",WINDOW_NORMAL);
           imshow("OriginalImage", img);
   }

    else if (parser.get<int>("@img_number") == 42 )
    {
        Mat padded=pad(img);
        Mat cmp=frequency_spectrum(padded);
        dftshift(cmp);


        Mat filter;
           filter = butter_lowpass(500, 3, cmp.size());
          /*  if (parser.has("lowpass")) {
                filter = butter_lowpass(250, 2, complex.size());
            } else {
                filter = butter_highpass(250, 2, complex.size());
            }
            */

                 Mat filtered=apply_filter(cmp,filter,img);
                 namedWindow("Filtered image",WINDOW_NORMAL);
                 imshow("Filtered image", filtered);



        // Visualize
            namedWindow("Input",WINDOW_NORMAL);
            imshow("Input", img);
/*

        vector<Rect> ROIs;
        namedWindow("ROI", WINDOW_NORMAL);
        selectROIs("ROI", mag_copy, ROIs);

       if(ROIs.size()<1)
             return 0;
        for (size_t i = 0; i < ROIs.size(); i++)
        {
            rectangle(magnitude, ROIs[i], Scalar(0), CV_FILLED);
        }

        show_frequency_spectrum(magnitude, "Magnitude_cropped");
       filtered = magnitude_to_img(magnitude, phase, planes, img);
        namedWindow("FilteredImage",WINDOW_NORMAL);
        cv::normalize(filtered, filtered, 0, 1, cv::NORM_MINMAX);
        imshow("FilteredImage", filtered);
        namedWindow("OriginalImage",WINDOW_NORMAL);
        imshow("OriginalImage", img);
   }

    // Display Cropped Image
    // namedWindow("Imagecr",WINDOW_AUTOSIZE);
    //  imshow("Imagecr", imCrop);
     // Show the image



    //namedWindow("Median FIlter",WINDOW_NORMAL);
    //imshow("Median FIlter", median);



    // Wait for escape key press before returning
        */
    while (waitKey() != 27);


    return 0;
}
}
