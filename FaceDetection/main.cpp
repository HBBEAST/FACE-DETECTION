#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

// ================= FACE DETECTION FUNCTION =================
void detectFaces(Mat& frame, CascadeClassifier& faceCascade) {
    Mat gray;
    vector<Rect> faces;

    // Convert to grayscale
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);

    // Detection parameters (no magic numbers)
    const double scaleFactor = 1.1;
    const int minNeighbors = 5;
    const Size minSize(30, 30);

    faceCascade.detectMultiScale(
        gray,
        faces,
        scaleFactor,
        minNeighbors,
        0,
        minSize
    );

    // Draw rectangles + label
    for (const auto& face : faces
