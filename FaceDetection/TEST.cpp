#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

// Function to detect faces in a frame
void detectFaces(Mat& frame, CascadeClassifier& faceCascade) {
    Mat gray;
    vector<Rect> faces;

    cvtColor(frame, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);

    faceCascade.detectMultiScale(
        gray,
        faces,
        1.1,
        5,
        0,
        Size(30, 30)
    );

    // Draw rectangles
    for (auto& face : faces) {
        rectangle(frame, face, Scalar(0, 255, 0), 2);
    }

    // Display face count
    string faceCountText = "Faces: " + to_string(faces.size());
    putText(frame, faceCountText, Point(20, 40),
        FONT_HERSHEY_SIMPLEX, 1.0,
        Scalar(0, 0, 255), 2);
}

int main() {

    CascadeClassifier faceCascade;
    if (!faceCascade.load("haarcascade_frontalface_default.xml")) {
        cout << "Error loading face cascade!" << endl;
        return -1;
    }

    cout << "Select Mode:\n";
    cout << "1 - Webcam Detection\n";
    cout << "2 - Image File Detection\n";
    cout << "Enter choice: ";

    int choice;
    cin >> choice;

    if (choice == 1) {
        VideoCapture cap(0);
        if (!cap.isOpened()) {
            cout << "Error opening webcam!" << endl;
            return -1;
        }

        Mat frame;
        double fps;
        int64 start;

        while (true) {
            start = getTickCount();

            cap >> frame;
            if (frame.empty()) break;

            detectFaces(frame, faceCascade);

            // FPS Calculation
            fps = getTickFrequency() / (getTickCount() - start);
            string fpsText = "FPS: " + to_string((int)fps);
            putText(frame, fpsText, Point(20, 80),
                FONT_HERSHEY_SIMPLEX, 1.0,
                Scalar(255, 0, 0), 2);

            imshow("Webcam Face Detection", frame);

            if (waitKey(10) == 27) break;
        }

        cap.release();
    }

    else if (choice == 2) {
        string imagePath;
        cout << "Enter image path: ";
        cin >> imagePath;

        Mat image = imread(imagePath);
        if (image.empty()) {
            cout << "Could not open image!" << endl;
            return -1;
        }

        detectFaces(image, faceCascade);

        imshow("Image Face Detection", image);
        waitKey(0);
    }

    destroyAllWindows();
    return 0;
}
