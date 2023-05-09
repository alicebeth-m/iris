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

from PIL import Image, ImageDraw

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
    pupilImg = cv2.inRange(frame, 10, 110)
    
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

def bilinear_interpolate(img, x, y):
    x1, y1 = np.floor(x).astype(np.uint8), np.floor(y).astype(np.uint8)
    x2, y2 = x1+1, y1+1
    dx, dy = x-x1, y-y1
    
    #if x1 < 0 or y1 < 0 or y2 >= img.shape[1] or x2 >= img.shape[0]:
        #return 0
    
    # Получаем значения пикселей в четырех ближайших точках
    p11 = img[x1, y1]
    p12 = img[x2, y1]
    p21 = img[x1, y2]
    p22 = img[x2, y2]
    
    # Выполняем билинейную интерполяцию
    result = (1-dx)*(1-dy)*p11 + dx*(1-dy)*p21 + (1-dx)*dy*p12 + dx*dy*p22
    return result
    
def integrate_arc(image, center, measure_arc, r):
    rows, cols = image.shape
    num1 = np.round(64*(measure_arc[3] + measure_arc[2])/90).astype(np.uint8)
    num2 = np.round(64*(measure_arc[0] + measure_arc[1])/90).astype(np.uint8)
    measure_arc = np.radians(measure_arc)
    angles_right = np.linspace(-measure_arc[3], measure_arc[2], num1) 
    x_right = np.array(center[1] + r * np.cos(angles_right))
    y_right = np.array(center[0] - r * np.sin(angles_right))    
    
    """filename = 'C:/Users/user/Downloads/iris/arcs/image_{}.jpg'.format(it)
    filepath = os.path.join('images_', filename)
    cv2.imwrite(filepath, draw)"""
    
    #measure_arc_0 = 180 - measure_arc[0]
    angles_left = np.linspace(np.pi - measure_arc[0], np.pi + measure_arc[1], num2)
    #print(angles_left)

    x_left = np.array(center[1] + r * np.cos(angles_left))
    y_left = np.array(center[0] - r * np.sin(angles_left))
    
    x = np.concatenate((x_left, x_right))
    y = np.concatenate((y_left, y_right))
   
    
    #print (max(x), max(y))
    """
    all_x = np.concatenate((x, x+1, x-1,x,x))
    all_y = np.concatenate((y, y, y, y+1, y-1))
    
    points = np.concatenate((all_x[None, :], all_y[None, :]), axis=0)
    uniq_points = np.unique(points, axis=1)
    #print (uniq_points)
    """
    
    if np.any(x >= rows - 1) or np.any(y >= cols - 1) or np.any(x <= 1) or np.any(y <= 1):
        # Этот процесс возвращает L=0 для любого круга, который не помещается внутри изображения
        return 0
    points_1 = bilinear_interpolate(image, y_left, x_left)
    points_2 = bilinear_interpolate(image, y_right, x_right)
    p1 = np.sum(points_1) / (num1 * r) 
    p2 = np.sum(points_2) /(num2 * r)
    
    #points = bilinear_interpolate(image, y, x)
    """
    draw = image.copy()
    for i in range(len(points_1)):
        cv2.circle(draw, (int(x_left[i]), int(y_left[i])), 1, (0, 0, 255), -1)
    for i in range(len(points_2)):
        cv2.circle(draw, (int(x_right[i]), int(y_right[i])), 1, (0, 0, 255), -1)
    #cv2.imshow('draw', draw)
    #cv2.waitKey(0)
    
    filename = 'C:/Users/user/Downloads/iris/arcs/image_{}.jpg'.format(it)
    filepath = os.path.join('images_', filename)
    cv2.imwrite(filepath, draw)
    """
    #print (np.sum(points)/(64 * r), p1+p2, num1, num2)
    #uniq_points = np.unique(points)
    return p1 + p2


def gauss_kernel(x, mu=0, sigma=15):
    return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x-mu)**2 / (2 * sigma**2))
def gaussian_der(r, sigma = 10):
    return (-r * np.exp(-(r**2)/(2*sigma**2)))/(sigma**3 * np.sqrt(2*np.pi))

def find_iris(image, y, x, radius):
    gray = image.copy()
    center = (y, x)
    h, w = gray.shape
    max_rad = int(min(x*0.9, (w-x)*0.9))
    area = []
    min_rad = int(radius*1.25)
    #gray = cv2.copyMakeBorder(gray, max_rad, max_rad, max_rad, max_rad, cv2.BORDER_CONSTANT, value=0)
    
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
    #print (measure_arc)
    """
    x1 = int(x-min_rad + 100 * np.cos(np.radians(180 - measure_arc[0])))
    y1 = int(y - 100 * np.sin(np.radians(180 - measure_arc[0])))
    x2 = int(x-min_rad + 100 * np.cos(np.radians(180 + measure_arc[1])))
    y2 = int(y - 100 * np.sin(np.radians(180 + measure_arc[1])))
    x3 = int(x+min_rad+1 + 100 * np.cos(np.radians(measure_arc[2])))
    y3 = int(y  - 100 * np.sin(np.radians(measure_arc[2])))
    x4 = int(x+min_rad+1 + 100 * np.cos(np.radians(-measure_arc[3])))
    y4 = int(y - 100 * np.sin(np.radians(-measure_arc[3])))
    img = image.copy()
    cv2.line(img, (x-min_rad, y), (x1, y1), (255, 0, 0), thickness=1)
    cv2.line(img, (x-min_rad, y), (x2, y2), (255, 0, 0), thickness=1)
    cv2.line(img, (x+min_rad+1, y), (x3, y3), (255, 0, 0), thickness=1)
    cv2.line(img, (x+min_rad+1, y), (x4, y4), (255, 0, 0), thickness=1)
    cv2.rectangle(img, (x-max_rad, y-min_rad), (x-min_rad, y), (0, 255, 0), 1)
    cv2.rectangle(img, (x-max_rad, y), (x-min_rad, y+min_rad), (0, 255, 0), 1)
    cv2.rectangle(img, (x+min_rad+1, y-min_rad), (x+max_rad, y), (0, 255, 0), 1)
    cv2.rectangle(img, (x+min_rad+1, y), (x+max_rad, y+min_rad), (0, 255, 0), 1)
    cv2.imshow("img",img)
    cv2.waitKey(0)
    """
    #max_rad = min(max_rad, r*3.75)
    n_delta_r = np.arange(int(min_rad + 2), int(max_rad))
    delta_r = 1
    k_shifts = np.array([-1, 0, 1])
    results = []
   
    g = gauss_kernel((k_shifts) * delta_r) - gauss_kernel((k_shifts-1) * delta_r)
    vectorize_integrate = np.vectorize(integrate_arc, excluded={0, 1, 2})
    all_rad = n_delta_r[:, None] +  k_shifts[None, :]
    #all_rad = np.arange(min_rad, max_rad)
    #print ("all_rad:", all_rad)
    if len(all_rad)* len(measure_arc) * len(center) > 0:
        t = vectorize_integrate(gray, center, measure_arc, all_rad) 
    else:
        return 0, 0, x, y
        
    results = (t @ g)
    result_shifts = results[1:] - results[:-1]
    '''
    Берём максимальную разность соседних по модулю
    '''
    #sigma = [0.5, 1, 5, 15]
    #sigma = np.array(sigma)
    #vect_gaus = np.vectorize(gaussian_der)
    #g = vect_gaus(all_rad)
    #t1 = np.ravel(t)
    #g1 = np.ravel(g)

    #result = np.abs(np.convolve(t1, g1, 'same'))
    idx = np.argmax(result_shifts)

    #print(all_rad[idx])
    #print(idx, max(result))

    #print("{}/{}: {}".format(idx, len(results), results[idx]))

    return int(n_delta_r[idx]), (results[idx]), x, y
    
    
    
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
    dist = 10
    t = image[x-r-dist:x+r+dist, y-r-dist:y+r+dist]
    img64_float = t.astype(np.float64)
    Mvalue = np.sqrt(((img64_float.shape[0]/2.0)**2.0)+((img64_float.shape[1]/2.0)**2.0))
    #print(t)
    thresh = cv2.inRange(t, 0, 80)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    """        
    pupilImg = image[x-r-dist:x+r+dist, y-r-dist:y+r+dist]
    for contour in contours:
        moments = cv2.moments(contour)
        area = moments['m00']
        if area > 50:
            x = moments['m10'] / area
            y = moments['m01'] / area
            pupil = contour
            cv2.drawContours(pupilImg, [pupil], -1, (0, 0, 0), -1)
            break        
    """
    #mask[x-r-dist:x+r+dist, y-r-dist:y+r+dist] = (cv2.Canny(mask[x-r-dist:x+r+dist, y-r-dist:y+r+dist],150,230))
    #image[x-r-dist:x+r+dist, y-r-dist:y+r+dist] = image[x-r-dist:x+r+dist, y-r-dist:y+r+dist] | mask[x-r-dist:x+r+dist, y-r-dist:y+r+dist]
    
    return max_contour

key = 0
#D:/CASIA-IrisV4-Interval/**/*[0-9].jpg
"""
x = np.arange(-10, 10, 0.1)

# Вычисление значений функции для каждого значения x
y = gaussian_der(x)

# Построение графика
plt.plot(x, y)

# Отображение графика
plt.show()
"""
global it 
it=0
#"D:/CASIA-IrisV4-Interval/**/*[0-9].jpg"
for name in glob.glob("D:/CASIA-IrisV4-Interval/**/*[0-9].jpg", recursive=True):
    iris = cv2.imread(name, 1)
    frame = iris.copy()
    kernel = np.ones((3,3),np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    
    
    erosion = cv2.erode(frame,kernel,iterations = 3)
    frame = cv2.dilate(erosion,kernel,iterations = 2)

    """ Проверка интерполяции
    new_img = np.zeros((gray.shape[0]*2, gray.shape[1]*2), dtype=np.uint8)
    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            y = j/2
            x = i/2
            new_img[i,j] = bilinear_interpolate(gray, x, y)
        
    cv2.imshow('new_image', new_img)
    cv2.waitKey(0)
    """
  
    """ Повышение контрастности """
    
    
    h, w = frame.shape
   
    gray_img = cv2.cvtColor(iris, cv2.COLOR_BGR2GRAY)
    mask = gray_img[int(h*0.2):int(h - h/4), int(w/4):int(w-w/4)]
    mean_intensity = cv2.mean(mask)[0]
    #cv2.imshow("m", mask)
    #cv2.waitKey(0)
    if mean_intensity < 90:
        brightness = 80
        frame = cv2.addWeighted(frame, 1, frame, 0, brightness)    
        frame = Image.fromarray(np.uint8(frame))
        img = Image.fromarray(np.uint8(frame))
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
        frame = np.array(img)    
        
    """
    brightness = 100
    frame = cv2.addWeighted(frame, 1, frame, 0, brightness)    
    frame = Image.fromarray(np.uint8(frame))
    enhancer = ImageEnhance.Contrast(frame)
    frame = enhancer.enhance(2)
    """
    circle = get_circle(gray_img[int(h*0.2):h, int(w/7):int(w-w/5)]) 
    
    centroid_pupil = (int(circle[0][0][0]) + int(w/7)), int(circle[0][0][1]) + int(h*0.2)
    #print ('x:', circle[0][0][0], 'y:', circle[0][0][1], 'r:', circle[0][0][2], "mean_intens:", mean_intensity)
    r = int(circle[0][0][2]) 
    
    x = np.arange(int(centroid_pupil[0] - 5), int(centroid_pupil[0] + 5))
    y = np.arange(int(centroid_pupil[1] - 5), int(centroid_pupil[1] + 5))
    xx, yy = np.meshgrid(x, y)
    xx, yy = np.round(xx/2).astype(np.uint8), np.round(yy/2).astype(np.uint8)
   
    vectorize_find_iris = np.vectorize(find_iris, excluded={0, 3})
    all_rad, max_grad, y1, x1 = vectorize_find_iris(gray, yy, xx, int(r/2)) 
    all_rad, max_grad, y1, x1 = all_rad.flatten(), max_grad.flatten(), y1.flatten(), x1.flatten()
    idx = np.argmax(max_grad)
    rad_ir = all_rad[idx] * 2
    #print(rad_ir)
    centroid = (y1[idx] * 2, x1[idx] * 2) 
    
    
    """
    max_contour = pupil_canny(frame, centroid_pupil, r)
    y, x = centroid_pupil
    dist = 10
    t = iris[x-r-dist:x+r+dist, y-r-dist:y+r+dist]
    cv2.drawContours(t, [max_contour], 0, (255,255,255), 1)
    """
    cv2.circle(iris, centroid, rad_ir, (87, 4, 89), 2)
    #iris = getIris(output)
    #cv2.imshow("input", frame)
    #cv2.imshow("output", iris)
    #cv2.waitKey(0)"""
    cv2.circle(iris, centroid_pupil, int(circle[0][0][2]), (169, 245, 9), 2)

    filename = 'C:/Users/user/Downloads/iris/results/image_{}.jpg'.format(it)
    filepath = os.path.join('images_', filename)
    print(filepath, name)
    cv2.imwrite(filepath, iris)
    it+=1
    
cv2.destroyAllWindows()
