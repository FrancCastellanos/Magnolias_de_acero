import cv2
import numpy as np

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def dilate(frame):
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(frame, kernel, iterations=7)
    return dilated

def erode(frame):
    kernel = np.ones((7, 7), np.uint8)
    eroded = cv2.erode(frame, kernel, iterations=7)
    return eroded

def process_frame(frame):
    # dilate and then erode to reduce noise
    noise_reduced = erode(dilate(frame))

    # Convert the HSV colorspace
    hsv = cv2.cvtColor(noise_reduced, cv2.COLOR_BGR2HSV)

    # Lower red Hue range
    lower1 = np.array([0,100,0])
    upper1 = np.array([5,255,255])

    # Upper red Hue range
    lower2 = np.array([150,100,0])
    upper2 = np.array([179,255,255])

    # Threshold the HSV image to get only blue color
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    mask = cv2.bitwise_or(mask1, mask2)

    # Bitwise-AND mask and original image
    res = cv2.medianBlur(frame, 5)
    res = cv2.bitwise_and(res, res, mask=mask)

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    gray = erode(dilate((gray)))

    ret, thresh = cv2.threshold(gray,0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    num_fruits = 0
    for cnt in contours:
        num_fruits += 1
        cv2.drawContours(gray, [cnt],0,(255,0,0),1)

    print('Num fruits: {}'.format(num_fruits))

    cv2.imshow('Original image', image_resize(frame, 1024))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('Color Detector', image_resize(thresh, 1024))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('photos/13.jpg', cv2.IMREAD_COLOR)
process_frame(img)
