import cv2
import sys
import os


imgname = "PataChitraPuri_1.jpg"
image = cv2.imread(imgname)



gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gaussian Blurring to remove high frequency noise helping in
# Contour Detection
gray = cv2.GaussianBlur(gray, (5, 5), 0)
# Canny Edge Detection
edged = cv2.Canny(gray, 75, 200)

# cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.imwrite('Edged.jpg', edged)
# finding the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)



# Handling due to different version of OpenCV
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# Taking only the top 5 contours by Area
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]


for c in cnts:

    # Calculates a contour perimeter or a curve length
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.1 * peri, True)  # 0.02

    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    screenCnt = approx
    if len(approx) == 4:
        screenCnt = approx
        break

cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Boundary", image)
cv2.imwrite('Boundary-' + imgname[0:2] + '.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()