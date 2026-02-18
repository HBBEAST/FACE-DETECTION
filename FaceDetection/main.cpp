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
    for (const auto& face : faces) {
        rectangle(frame, face, Scalar(0, 255, 0), 2);

        putText(frame, "Face",
                Point(face.x, face.y - 5),
                FONT_HERSHEY_SIMPLEX,
                0.6,
                Scalar(0, 255, 0),
                2);
    }

    // Display face count
    string faceCountText = "Faces: " + to_string(faces.size());
    putText(frame, faceCountText,
            Point(20, 40),
            FONT_HERSHEY_SIMPLEX,
            1.0,
            Scalar(0, 0, 255),
            2);
}

// ============================ MAIN ============================
int main() {

    // 🔴 IMPORTANT: adjust path if needed
    const string cascadePath = "haarcascade_frontalface_default.xml";

    CascadeClassifier faceCascade;
    if (!faceCascade.load(cascadePath)) {
        cout << "❌ Error loading face cascade!" << endl;
        cout << "Make sure haarcascade_frontalface_default.xml is in working directory." << endl;
        return -1;
    }

    cout << "Select Mode:\n";
    cout << "1 - Webcam Detection\n";
    cout << "2 - Image File Detection\n";
    cout << "Enter choice: ";

    int choice;
    if (!(cin >> choice)) {
        cout << "❌ Invalid input!" << endl;
        return -1;
    }

    // ================= WEBCAM MODE =================
    if (choice == 1) {
        VideoCapture cap(0);

        if (!cap.isOpened()) {
            cout << "❌ Error opening webcam!" << endl;
            return -1;
        }

        Mat frame;
        int64 start;
        static double fpsSmooth = 0.0;

        cout << "Press ESC to exit webcam..." << endl;

        while (true) {
            start = getTickCount();

            cap >> frame;
            if (frame.empty()) break;

            // Optional resize (better performance)
            resize(frame, frame, Size(), 0.75, 0.75);

            detectFaces(frame, faceCascade);

            // ===== Smooth FPS calculation =====
            double currentFPS =
                getTickFrequency() / (getTickCount() - start);
            fpsSmooth = 0.9 * fpsSmooth + 0.1 * currentFPS;

            string fpsText = "FPS: " + to_string((int)fpsSmooth);
            putText(frame, fpsText,
                    Point(20, 80),
                    FONT_HERSHEY_SIMPLEX,
                    1.0,
                    Scalar(255, 0, 0),
                    2);

            imshow("Webcam Face Detection", frame);

            if (waitKey(10) == 27) break; // ESC key
        }

        cap.release();
    }

    // ================= IMAGE MODE =================
    else if (choice == 2) {
        string imagePath;
        cout << "Enter image path: ";
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        getline(cin, imagePath);


        Mat image = imread(imagePath);
        if (image.empty()) {
            cout << "❌ Could not open image!" << endl;
            return -1;
        }

        detectFaces(image, faceCascade);

        imshow("Image Face Detection", image);
        waitKey(0);
    }

    else {
        cout << "❌ Invalid choice!" << endl;
    }

    destroyAllWindows();
    return 0;
}
