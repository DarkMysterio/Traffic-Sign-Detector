#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

/* ----------  UTILITĂȚI  -------------------------------------------------- */

// 1. Pentru YIELD: verifică dacă triunghiul are vârful în sus și baza în jos
bool isTrianglePointingUp(const vector<Point>& tri) {
    Moments m = moments(tri);
    Point2f c(m.m10 / m.m00, m.m01 / m.m00);

    int above = 0, below = 0;
    for (auto& p : tri) (p.y < c.y) ? ++above : ++below;

    return above == 1 && below == 2;
}
// 1. Pentru WARNING: verifică dacă triunghiul are vârful în JOS și baza în Sus
bool isTrianglePointingDown(const vector<Point>& tri) {
    Moments m = moments(tri);
    Point2f c(m.m10 / m.m00, m.m01 / m.m00);

    int above = 0, below = 0;
    for (auto& p : tri) (p.y < c.y) ? ++above : ++below;

    return below == 1 && above == 2;
}

// 2. Returnează raportul (pixeli_roșii / pixeli_totali) în interiorul unui contur
double redFillRatio(const vector<Point>& poly,
                    const Mat& redMask,
                    const Size& size) {
    Mat polyMask(size, CV_8UC1, Scalar(0));
    vector<vector<Point>> tmp{poly};
    fillPoly(polyMask, tmp, Scalar(255));

    Mat redInside;
    bitwise_and(polyMask, redMask, redInside);

    double redPx   = countNonZero(redInside);
    double totalPx = countNonZero(polyMask);
    return (totalPx == 0) ? 0.0 : redPx / totalPx;
}

/* ------------------------------------------------------------------------ */

int main() {
    // (1) ÎNCĂRCARE IMAGINE
    Mat src = imread("P:\\CURSURI\\pi\\lab_code\\proiect\\poze\\warning.png");
    if (src.empty()) {
        cerr << "Eroare la deschiderea imaginii!" << endl;
        return -1;
    }

    // (2) BLUR + HSV + SEGMENTARE ROȘU
    Mat blurred, hsv;
    GaussianBlur(src, blurred, Size(5,5), 2);
    cvtColor(blurred, hsv, COLOR_BGR2HSV);

    Mat mask1, mask2, redMask;
    inRange(hsv, Scalar(  0,  60,  80), Scalar( 10,255,255), mask1);
    inRange(hsv, Scalar(160,  60,  80), Scalar(180,255,255), mask2);
    redMask = mask1 | mask2;

    morphologyEx(redMask, redMask, MORPH_CLOSE,
                 getStructuringElement(MORPH_ELLIPSE, Size(5,5)));
    morphologyEx(redMask, redMask, MORPH_OPEN,
                 getStructuringElement(MORPH_ELLIPSE, Size(3,3)));

    // (3) DETECTARE CONTURURI
    vector<vector<Point>> contours;
    findContours(redMask, contours, noArray(),
                 RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // (4) CLASIFICARE
    Mat output = src.clone();
    int fontFace  = FONT_HERSHEY_SIMPLEX;
    double fScale = 0.9;
    int fThick    = 2;

    for (auto& c : contours) {
        double area = contourArea(c);
        if (area < 500) continue;

        vector<Point> hull;
        convexHull(c, hull);


        double peri = arcLength(hull, true);
        vector<Point> approx;
        approxPolyDP(hull, approx, 0.04 * peri, true);

        Moments m = moments(c);
        Point2f cent(m.m10/m.m00, m.m01/m.m00);

        double circularity = 4 * CV_PI * area / (peri * peri);
        double redRatio    = redFillRatio(approx, redMask, src.size());

        string label;
        Scalar color;
        bool matched = false;

        // (a) YIELD: triunghi inversat
        if (approx.size() == 3 && isTrianglePointingDown(approx)) {
            label   = "YIELD";
            color   = Scalar(0,255,0);
            matched = true;
        }
        else if (approx.size() == 3 &&
                 isTrianglePointingUp(approx) &&
                 redRatio > 0.15 && redRatio < 0.60) {
            label   = "WARNING";
            color   = Scalar(0, 200, 255);
            matched = true;
        }
            // (b) STOP: 8 varfuri predominant rosu
        else if (approx.size() == 8
                 && isContourConvex(approx)
                 && redRatio > 0.85) {
            label   = "STOP";
            color   = Scalar(0,255,0);
            matched = true;
        }
            // (c) Semne circulare (circularitate ridicată)
        else if (circularity > 0.80) {
            if (redRatio < 0.35) {
                // SPEED limit: zonă albă în interior
                label   = "SPEED";
                color   = Scalar(255,0,0);
                matched = true;
            }
            else {
                // NO ENTRY: redRatio moderat spre înalt
                label   = "NO ENTRY";
                color   = Scalar(0,0,255);
                matched = true;
            }
        }

        // desen și text
        if (matched) {
            polylines(output, approx, true, color, 3);
            int base=0;
            Size ts = getTextSize(label, fontFace, fScale, fThick, &base);
            Point org(int(cent.x - ts.width/2),
                      int(cent.y + ts.height/2));
            putText(output, label, org, fontFace, fScale, color, fThick);
        }
    }

    // (5) AFIȘARE
    imshow("Detected Signs", output);
    waitKey();
    return 0;
}