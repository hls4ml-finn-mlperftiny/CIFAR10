import numpy as np
import cv2 as cv

# for plotting / checking
# from google.colab.patches import cv2_imshow
# import matplotlib.pyplot as plt


def load_data():
    # original data from: https://cdn.shopify.com/s/files/1/1708/0909/products/wrapping_paper.jpg
    img = cv.imread('wrapping_paper.jpg', cv.IMREAD_GRAYSCALE)
    img = img[:800]
    img = cv.medianBlur(img, 5)
    cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20,
                              param1=50, param2=30, minRadius=96, maxRadius=97)
    circles = np.uint16(np.around(circles))[0]
    for i in circles:
        # draw the outer circle
        cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    # cv2_imshow(cimg)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    data = cv.imread('wrapping_paper.jpg', cv.IMREAD_COLOR)
    data = data[: 800, :, ::-1]  # swap BGR -> RGB
    X = np.zeros((36, 96 * 2, 96 * 2, 3), dtype=int)
    X_resize = np.zeros((36, 32, 32, 3), dtype=int)
    y = np.zeros((36, 36), dtype=int)

    circles = circles[np.lexsort((circles[:, 1], circles[:, 0]))]
    # fig, axs = plt.subplots(4, 9, figsize=(18, 8))
    for ic, circle in enumerate(circles):
        cen_x = circle[0]
        cen_y = circle[1]
        radius = circle[2]
        X[ic, :, :, :] = data[cen_y - radius: cen_y + radius,
                              cen_x - radius: cen_x + radius,
                              :]
        X_inter = X[ic, :, :, :].astype(float)
        X_inter = cv.pyrDown(X_inter, dstsize=(96, 96))
        X_inter = cv.pyrDown(X_inter, dstsize=(48, 48))
        X_resize[ic, :, :, :] = cv.resize(X_inter, dsize=(32, 32), interpolation=cv.INTER_LINEAR_EXACT).astype(int)
        k = 0
        found = False
        for i in range(0, 9):
            for j in range(0, 4):
                if cen_x > i * 200 and cen_x < (i + 1) * 200 and cen_y > j * 200 and cen_y < (j + 1) * 200:
                    y[ic, k] = 1
                    # axs[j, i].imshow(X_resize[ic])
                    found = True
                    break
                k += 1
            if found:
                break

    return (X_resize, y), (X_resize, y)
