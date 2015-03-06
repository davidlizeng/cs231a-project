# std::vector<cv::Rect> detectLetters(cv::Mat img)
# {
#     std::vector<cv::Rect> boundRect;
#     cv::Mat img_gray, img_sobel, img_threshold, element;
#     cvtColor(img, img_gray, CV_BGR2GRAY);
#     cv::Sobel(img_gray, img_sobel, CV_8U, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
#     cv::threshold(img_sobel, img_threshold, 0, 255, CV_THRESH_OTSU+CV_THRESH_BINARY);
#     element = getStructuringElement(cv::MORPH_RECT, cv::Size(17, 3) );
#     cv::morphologyEx(img_threshold, img_threshold, CV_MOP_CLOSE, element); //Does the trick
#     std::vector< std::vector< cv::Point> > contours;
#     cv::findContours(img_threshold, contours, 0, 1);
#     std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
#     for( int i = 0; i < contours.size(); i++ )
#         if (contours[i].size()>100)
#         {
#             cv::approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 3, true );
#             cv::Rect appRect( boundingRect( cv::Mat(contours_poly[i]) ));
#             if (appRect.width>appRect.height)
#                 boundRect.push_back(appRect);
#         }
#     return boundRect;
# }
import cv2
import cv
import numpy as np

def detectLetters(img):
    img_gray = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    img_lap = cv2.Laplacian(img_gray,cv2.CV_8U)
    #cv2.imwrite('lap.png', img_lap)
    #img_sobel = cv2.Sobel(img_gray, cv2.CV_8U, 0, 1, ksize=5)
    #cv2.imwrite('sobel.png', img_sobel)
    #img_sobel = cv2.Sobel(img_gray, cv2.CV_8U, 1, 0, 3, 1, 0, cv2.BORDER_DEFAULT)
    unused, img_threshold = cv2.threshold(img_lap, 0, 255, cv.CV_THRESH_OTSU + cv.CV_THRESH_BINARY)
    #cv2.imwrite('thresh.png', img_threshold)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 4) )
    morphed = cv2.morphologyEx(img_threshold, cv.CV_MOP_CLOSE, element)
    cv2.imwrite('morph.png', morphed)
    contours, unused = cv2.findContours(morphed, 0, 1)
    boundRect = []
    for i in xrange(len(contours)):
        if len(contours[i]) > 10:
            contour_poly = cv2.approxPolyDP(contours[i], 3, True)
            appRect = cv2.boundingRect(contour_poly)
            boundRect.append(appRect)
    return boundRect

img = cv2.imread('equation.png')
boundRect = detectLetters(img)
for rect in boundRect:
    cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1] + rect[3]), (0,255,0))
cv2.imwrite('eq-morph.png', img)
