import cv2 
import numpy as np
import math
import glob
from typing import List
from scipy import ndimage
from scipy.signal import convolve
from PIL import Image, ImageEnhance
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
# GLOBAL VARIABLES
#####################################
# Holds the pupil's center
centroid = (0,0)
# Holds the iris' radius
radius = 0
# Holds the current element of the image used by the getNewEye function
currentEye = 0
# Holds the list of eyes (filenames)
eyesList = []
#####################################
def get_circle(image):
    h, w = image.shape
    max_rad = int(w/2)
    #blur = cv2.GaussianBlur(image, (5, 5), 0)
    filterd_image  = cv2.medianBlur(image,7)

    filterd_image = cv2.Canny(filterd_image,150,230)
    
    circles = cv2.HoughCircles(filterd_image, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=30, param2=10, minRadius=10, maxRadius=max_rad)
    #circles = np.uint8(np.around(circles))
    # Step 6: Return the first detected circle (if any)
    if circles is not None:
        return circles
    else:
        return None

# Returns a different image filename on each call. If there are no more
# elements in the list of images, the function resets.
#
# @param list		List of images (filename)
# @return string	Next image (filename). Starts over when there are
#			no more elements in the list.
def getNewEye(list):
	global currentEye
	if (currentEye >= len(list)):
		currentEye = 0
	newEye = list[currentEye]
	currentEye += 1
	return (newEye)

# Returns the cropped image with the isolated iris and black-painted
# pupil. It uses the getCircles function in order to look for the best
# value for each image (eye) and then obtaining the iris radius in order
# to create the mask and crop.
#
# @param image		Image with black-painted pupil
# @returns image 	Image with isolated iris + black-painted pupil
def getIris(frame):
    iris = []
    copyImg = frame.copy()
    resImg = frame.copy()
    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(grayImg)
    edges = cv2.Canny(grayImg, 5, 70, 3)
    edges = cv2.GaussianBlur(edges, (7, 7), 0)
    circles = getCircles(edges)
    iris.append(resImg)
    for circle in circles:
        #centroid = (int(circle[0][0]), int(circle[0][1]))
        rad = int(circle[0][2])
        global radius
        radius = rad
        cv2.circle(mask, centroid, rad, (255, 255, 255), cv2.FILLED)
        mask = cv2.bitwise_not(mask)
        cv2.subtract(frame, copyImg, resImg, mask)
        x = int(centroid[0] - rad)
        y = int(centroid[1] - rad)
        w = int(rad * 2)
        h = w
        resImg1 = resImg[y:y+h, x:x+w]
        return resImg1
    return resImg

def getCircles(image):
    i = 80
    while i < 151:
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 2, 100.0, param1=30, param2=i, minRadius=100, maxRadius=140)
        if circles is not None and len(circles) == 1:
            return circles
        i += 1
    return []

# Returns the same images with the pupil masked black and set the global
# variable centroid according to calculations. It uses the FindContours 
# function for finding the pupil, given a range of black tones.

# @param image		Original image for testing
# @returns image	Image with black-painted pupil
def getPupil(frame):
    pupilImg = cv2.inRange(frame, (10, 10, 10), (110, 110, 110))
    
    contours, _ = cv2.findContours(pupilImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pupilImg = frame.copy()
    for contour in contours:
        moments = cv2.moments(contour)
        area = moments['m00']
        if area > 50:
            x = moments['m10'] / area
            y = moments['m01'] / area
            pupil = contour
            global centroid
            centroid = (int(x), int(y))
            cv2.drawContours(pupilImg, [pupil], -1, (0, 0, 0), -1)
            break
    return pupilImg


def integrate_arc(image, center, measure_arc, r):
    rows, cols = image.shape
    #512
    angles_right = np.linspace(-measure_arc[3], measure_arc[2]) / 180 * np.pi
    x_right = np.floor(center[0] + r * np.cos(angles_right)).astype(np.uint8)
    y_right = np.floor(center[1] - r * np.sin(angles_right)).astype(np.uint8)
    angles_left = np.linspace(180 - measure_arc[0], 180 + measure_arc[1]) / 180 * np.pi
    x_left = np.floor(center[0] + r * np.cos(angles_left)).astype(np.uint8)
    y_left = np.floor(center[1] - r * np.sin(angles_left)).astype(np.uint8)
    x = np.concatenate((x_left, x_right))
    y = np.concatenate((y_left, y_right))
    all_x = np.concatenate((x, x+1, x-1,x,x))
    all_y = np.concatenate((y, y, y, y+1, y-1))
    points = np.concatenate((all_x[None, :], all_y[None, :]), axis=0)
    uniq_points = np.unique(points, axis=1)
    #print (uniq_points)

    if np.any(x >= rows) or np.any(y >= cols) or np.any(x <= 1) or np.any(y <= 1):
        # Этот процесс возвращает L=0 для любого круга, который не помещается внутри изображения
        return 0
    return np.sum(image[uniq_points[0], uniq_points[1]]) / (2 * np.pi * r)


def gauss_kernel(x, mu=0, sigma=9):
    return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x-mu)**2 / (2 * sigma**2))


def find_iris(image, center, radius):
    gray = image.copy()
    y, x = center
    h, w = gray.shape
    max_rad = int(min(radius*3.75, w*2/5))
    area = []
    min_rad = int(radius*1.25)
    area.append(gray[y-min_rad:y, x-max_rad:x-min_rad])
    h1, w1 = area[0].shape
    area.append(gray[y:y+min_rad, x-max_rad:x-min_rad])
    area.append(gray[y-min_rad:y, x+min_rad+1:x+max_rad])
    area.append(gray[y:y+min_rad, x+min_rad+1:x+max_rad])
    areas_sum = 0
    measure_arc = []
    for i in range (4):
        measure_arc.append(90*np.sum(area[i]))
        areas_sum += np.sum(area[i])
    measure_arc /= areas_sum   
    #print("min_rad = {}, max_rad = {}".format(min_rad, max_rad))
    
    
    
    n_delta_r = np.arange(int(min_rad*1.1), int(max_rad*0.95))
    delta_r = 1
    k_shifts = np.array([-1, 0, 1])
    results = []
    g = gauss_kernel((k_shifts) * delta_r) - gauss_kernel((k_shifts-1) * delta_r)
    vectorize_integrate = np.vectorize(integrate_arc, excluded={0, 1, 2})
    all_rad = n_delta_r[:, None] +  k_shifts[None, :]
    if len(all_rad)* len(measure_arc) * len(center) > 0:
        t = vectorize_integrate(gray, center, measure_arc, all_rad) 
    else:
        return -1
    results = (t @ g)
    result_shifts = results[1:] - results[:-1]
    '''
    Берём максимальную разность соседних по модулю
    '''
    idx = np.argmax(result_shifts)
    #print("{}/{}: {}".format(idx, len(results), results[idx]))
    return int(n_delta_r[idx])
# Returns the image as a "tape" converting polar coord. to Cartesian coord.
#
# @param image		Image with iris and pupil
# @returns image	"Normalized" image
def getPolar2CartImg(image, rad):
    imgSize = image.shape
    c = (float(imgSize[0]/2.0), float(imgSize[1]/2.0))
    imgRes = np.zeros((int(360), rad*3, 3), dtype=np.uint8)
    #cv2.logPolar(image,imgRes,c,(50.0,cv2.CV_INTER_LINEAR+cv2.WARP_FILL_OUTLIERS))
    cv2.logPolar(image,imgRes,c,(60.0,cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS))
    return (imgRes)

def pupil_canny(image, center, r):
    y, x = center
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(image, (5, 5), 5)
    #mask = blur.copy()
    dist = int(r//4)
    t = image[x-r-dist:x+r+dist, y-r-dist:y+r+dist]
    img64_float = t.astype(np.float64)
    Mvalue = np.sqrt(((img64_float.shape[0]/2.0)**2.0)+((img64_float.shape[1]/2.0)**2.0))
    #ploar_image = cv2.linearPolar(img64_float,(img64_float.shape[0]/2, img64_float.shape[1]/2),Mvalue,cv2.WARP_FILL_OUTLIERS)
    #cartisian_image = cv2.linearPolar(ploar_image, (img64_float.shape[0]/2, img64_float.shape[1]/2),Mvalue, cv2.WARP_INVERSE_MAP)
    #cartisian_image = cartisian_image/200
    #kernel = np.ones((5,5),np.uint8)
    #cartisian_image = cv2.erode(cartisian_image,kernel,iterations = 1)
    #ploar_image = cv2.Canny(ploar_image.astype(np.uint8),150,230)
    #ploar_image = ploar_image/255
    cartisian_image = t.astype(np.uint8)
    thresh = cv2.inRange( t, 0, 80 )
  
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = None
    max_area = 0
    i = 0
    i_max = 0
    for contour in contours:
        i+=1
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
            i_max = i

    #cv2.drawContours(t, [max_contour], 0, (255,255,255), 1)
    #cv2.imshow('Max Contour', t)
    #cv2.drawContours(t, contours, 0, (255, 0, 0), 1, cv2.LINE_AA, hierarchy, 1)
    #cv2.imshow("log-polar1", ploar_image)
    #cv2.imshow("log-polar2", t)
     
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    """
    mask[x-r-dist:x+r+dist, y-r-dist:y+r+dist] = (cv2.Canny(mask[x-r-dist:x+r+dist, y-r-dist:y+r+dist],150,230))
    image[x-r-dist:x+r+dist, y-r-dist:y+r+dist] = image[x-r-dist:x+r+dist, y-r-dist:y+r+dist] | mask[x-r-dist:x+r+dist, y-r-dist:y+r+dist]
    """
    return max_contour

key = 0
i = 0

for name in glob.glob("D:/CASIA-IrisV4-Interval/**/*[0-9].jpg", recursive=True):
    iris = cv2.imread(name, 1)
    frame = iris.copy()
    kernel = np.ones((5,5),np.uint8)
    #frame = cv2.GaussianBlur(iris, (7, 7), 5)
    
    erosion = cv2.erode(frame,kernel,iterations = 3)
    frame = cv2.dilate(erosion,kernel,iterations = 2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    """
    img = Image.fromarray(np.uint8(gray))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)
    img = np.array(img)
    """
    img = gray
    
    circle = get_circle(img)
    """if (circle is None):
        filename = 'C:/Users/user/Downloads/iris/res_2/image_{}.jpg'.format(i)
        filepath = os.path.join('images_', filename)
        #iris = np.zeros(iris.shape, dtype=np.uint8)
        plt.imsave(filepath, iris)
        i+=1
        continue
        """
    centroid = (int(circle[0][0][0]), int(circle[0][0][1])) 
    rad_ir = find_iris(img, centroid, int(circle[0][0][2]))
    """if rad_ir == -1:
        filename = 'C:/Users/user/Downloads/iris/res_2/figure_{}.jpg'.format(i)
        filepath = os.path.join('images_', filename)
        iris = np.zeros(iris.shape, dtype=np.uint8)
        plt.imsave(filepath, iris)
        i+=1
        continue
    """
    img1 = cv2.erode(img,kernel,iterations = 3)
    cv2.imshow("output", img1)
    cv2.waitKey(0)
    max_contour = pupil_canny(img1, centroid, int(circle[0][0][2]))
    y, x = centroid
    r = int(circle[0][0][2])
    dist = int(r//4)
    t = iris[x-r-dist:x+r+dist, y-r-dist:y+r+dist]
    cv2.drawContours(t, [max_contour], 0, (255,255,255), 1)
    cv2.circle(iris, centroid, rad_ir, (87, 160, 0), 2)
    #cv2.circle(iris, centroid, int(circle[0][0][2]), (169, 245, 244), 2)
    #iris = getIris(output)
    #cv2.imshow("input", frame)
    #cv2.imshow("output", iris)
    #cv2.waitKey(0)
    filename = 'C:/Users/user/Downloads/iris/results/figure_{}.jpg'.format(i)
    filepath = os.path.join('images_', filename)
    plt.imsave(filepath, iris)
    
    i+=1
    
cv2.destroyAllWindows()
