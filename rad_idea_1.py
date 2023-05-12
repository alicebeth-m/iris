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
    
def integrate_arc(image, center, angle_start, angle_end, num, r):
    rows, cols = image.shape
    
    angles = np.linspace(angle_start, angle_end, num) 
    x = np.array(center[1] + r * np.cos(angles))
    y = np.array(center[0] - r * np.sin(angles))    

    if np.any(x >= rows - 1) or np.any(y >= cols - 1) or np.any(x <= 1) or np.any(y <= 1):
        # Этот процесс возвращает L=0 для любого круга, который не помещается внутри изображения
        return 0
    points = bilinear_interpolate(image, y, x)
    """
    draw = image.copy()
    for i in range(len(points_1)):
        cv2.circle(draw, (int(x_left[i]), int(y_left[i])), 1, (0, 0, 255), -1)
    for i in range(len(points_2)):
        cv2.circle(draw, (int(x_right[i]), int(y_right[i])), 1, (0, 0, 255), -1)
    cv2.imshow('draw', draw)
    cv2.waitKey(0)
    
    filename = 'C:/Users/user/Downloads/iris/arcs/image_{}.jpg'.format(it)
    filepath = os.path.join('images_', filename)
    cv2.imwrite(filepath, draw)
    """
    #print (np.sum(points)/(64 * r), p1+p2, num1, num2)
    #uniq_points = np.unique(points)
    return np.sum(points) / (num *  r)


def gauss_kernel(x, mu=0, sigma=20):
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
    num1 = np.round(128*(measure_arc[3] + measure_arc[2])/90).astype(np.uint8)

    measure_arc = np.radians(measure_arc)
    
    num2 = 128 - num1

    measure_arc[0] = np.pi - measure_arc[0]
    measure_arc[1] = np.pi + measure_arc[1]
    measure_arc[3] = -measure_arc[3]
    
    results_left = []
    results_right = []
    max_rad = min(max_rad, r*5)
    vectorize_integrate = np.vectorize(integrate_arc, excluded={0, 1, 2, 3, 4})
    all_rad = np.arange(min_rad, max_rad)
    if len(all_rad)* len(measure_arc) * len(center) > 0:
        t_left = vectorize_integrate(gray, center, measure_arc[0], measure_arc[1], num2, all_rad) 
        t_right = vectorize_integrate(gray, center, measure_arc[3], measure_arc[2], num1, all_rad) 
    else:
        return 0, 0, x, y
    sigma = np.arange(1, 10, 1)
    vect_gaus = np.vectorize(gauss_kernel)
    for s in sigma:
        g = vect_gaus(all_rad, s)    
        results_left.append(np.abs(np.convolve(t_left, g)))
        results_right.append(np.abs(np.convolve(t_right, g)))
        
    results_left = np.array(results_left)
    results_right = np.array(results_right)
    result_shifts_left = results_left[1:] - results_left[:-1]
    result_shifts_right = results_right[1:] - results_right[:-1]

    max_val_left = np.amax(result_shifts_left)
    max_val_right = np.amax(result_shifts_right)
    
    idx_left = np.unravel_index(np.argmax(result_shifts_left), result_shifts_left.shape)
    idx_right = np.unravel_index(np.argmax(result_shifts_right), result_shifts_right.shape)
    
    radius = int((all_rad[idx_left[1]] + all_rad[idx_right[1]])/2)
    max_val = int((max_val_left + max_val_right)/2)
    #idx = np.argmax(results)

    #print(all_rad[idx])
    #print(idx, max(result))

    #print("{}/{}: {}".format(idx, len(results), results[idx]))

    return radius, (max_val), x, y
    
    

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
        img = enhancer.enhance(2)
        frame = np.array(img)    
    else:    
        brightness = 50
        gray = cv2.addWeighted(gray, 1, gray, 0, brightness)    
        gray = Image.fromarray(np.uint8(gray))
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(1.5)
        gray = np.array(gray)
    """
    cv2.imshow('gra', gray)
    cv2.waitKey(0)"""
    circle = get_circle(gray_img[int(h*0.2):h, int(w/7):int(w-w/5)]) 
    
    centroid_pupil = (int(circle[0][0][0]) + int(w/7)), int(circle[0][0][1]) + int(h*0.2)
    #print ('x:', circle[0][0][0], 'y:', circle[0][0][1], 'r:', circle[0][0][2], "mean_intens:", mean_intensity)
    r = int(circle[0][0][2]) 
    
    x = np.arange(int(centroid_pupil[0] - 10), int(centroid_pupil[0] + 10))
    y = np.arange(int(centroid_pupil[1] + 2), int(centroid_pupil[1] + 7))
    xx, yy = np.meshgrid(x, y)
    xx, yy = np.round(xx/2).astype(np.uint8), np.round(yy/2).astype(np.uint8)
   
    vectorize_find_iris = np.vectorize(find_iris, excluded={0, 3})
    all_rad, max_grad, y1, x1 = vectorize_find_iris(gray, yy, xx, int(r/2)) 
    all_rad, max_grad, y1, x1 = all_rad.flatten(), max_grad.flatten(), y1.flatten(), x1.flatten()
    idx = np.argmax(max_grad)
    rad_ir_left = all_rad[idx] * 2
    #print(rad_ir)
    centroid_left = (y1[idx] * 2, x1[idx] * 2) 
    
    all_rad, max_grad, y1, x1 = vectorize_find_iris(gray, yy, xx, int(r/2)) 
    all_rad, max_grad, y1, x1 = all_rad.flatten(), max_grad.flatten(), y1.flatten(), x1.flatten()
    idx = np.argmax(max_grad)
    rad_ir_right = all_rad[idx] * 2
    #print(rad_ir)
    centroid_right = (y1[idx] * 2, x1[idx] * 2) 
    
    
    """
    max_contour = pupil_canny(frame, centroid_pupil, r)
    y, x = centroid_pupil
    dist = 10
    t = iris[x-r-dist:x+r+dist, y-r-dist:y+r+dist]
    cv2.drawContours(t, [max_contour], 0, (255,255,255), 1)
    """
    cv2.circle(iris, centroid, int(rad_ir), (87, 4, 89), 2)
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
