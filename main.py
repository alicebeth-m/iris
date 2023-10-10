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


#-------------------PUPIL-------------------#

def get_circle(image):
    h, w = image.shape
    max_rad = int(w/2)
    #filterd_image  = cv2.medianBlur(image,7)
    filterd_image = cv2.bilateralFilter(image, 9, 75, 75)
    #show_image(filterd_image)

    filterd_image = cv2.Canny(filterd_image,150,230)
    circles = cv2.HoughCircles(filterd_image, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=30, param2=10, minRadius=10, maxRadius=max_rad)
    if circles is not None:
        return circles
    else:
        return None

def pupil_search(frame):
    kernel = np.ones((3,3),np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = frame.shape
    gray_img = cv2.cvtColor(iris, cv2.COLOR_BGR2GRAY)
    mask = gray_img[int(h*0.2):int(h - h/4), int(w/4):int(w-w/4)]
    mean_intensity = cv2.mean(mask)[0]
    print(mean_intensity)
    erosion = cv2.erode(frame,kernel,iterations = 3)
    frame = cv2.dilate(erosion,kernel,iterations = 2)
    brightness = 80
    frame = cv2.addWeighted(frame, 1, frame, 0, brightness)    
    frame = Image.fromarray(np.uint8(frame))
    img = Image.fromarray(np.uint8(frame))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)
    frame = np.array(img) 
        
    circle = get_circle(frame[int(h*0.2):h, int(w/7):int(w-w/5)]) 
    centroid_pupil = (int(circle[0][0][0]) + int(w/7)), int(circle[0][0][1]) + int(h*0.2)
    r = int(circle[0][0][2]) 
    return centroid_pupil, r
    
#-------------------IRIS-------------------#

#Обработка изображения радужки
def image_preprocessing(gray, param):
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)   
    resize_param = 1/param
    gray = cv2.resize(gray, None, fx=resize_param, fy=resize_param, interpolation=cv2.INTER_AREA)
    return gray 



#Билинейная интерполяция
def bilinear_interpolate(img, x, y):
    x1, y1 = np.floor(x).astype(np.uint8), np.floor(y).astype(np.uint8)
    x2, y2 = x1+1, y1+1
    dx, dy = x-x1, y-y1
    #if np.any(x1 < 0) or np.any(y1 < 0) or np.any(y2 >= img.shape[0]) or np.any(x2 >= img.shape[1]):
        #return 0
    # Получаем значения пикселей в четырех ближайших точках
    p11 = img[y1, x1]
    p12 = img[y2, x1]
    p21 = img[y1, x2]
    p22 = img[y2, x2]    
    # Выполняем билинейную интерполяцию
    result = (1-dx)*(1-dy)*p11 + dx*(1-dy)*p21 + (1-dx)*dy*p12 + dx*dy*p22
    return result


#Подсчет кругового интеграла    
def integral(image, center, angle_start, angle_end, r):
    rows, cols = image.shape
    angles = np.linspace(angle_start, angle_end, int(r*abs(angle_end - angle_start))) 
    x = np.array(center[0] + r * np.cos(angles))
    y = np.array(center[1] - r * np.sin(angles))   

    
    
    if np.any(y >= rows - 1) or np.any(x >= cols - 1) or np.any(y <= 1) or np.any(x <= 1):
        # Этот процесс возвращает L=0 для любого круга, который не помещается внутри изображения
        return 0
    points = bilinear_interpolate(image, x, y)
    integral = np.sum(points) / (r*abs(angle_end - angle_start))
    """
    #print(f"integral={integral}")       
    plt.imshow(image)
    # отображение массивов x и y на изображении
    plt.plot(x, y, 'r')
    plt.text(center[0], center[1], f'integral: {integral}°', fontsize=12)
    # отображение графика
    plt.show()
    """
    return integral 
    

#Ядро Гаусса    
def gaussian_kernel(size, sigma=10, dimension=1):
    gk = np.empty([size])
    for i in range(size):
        gk[i] = (1/(((2*np.pi)**0.5)*sigma))*np.exp(-((i-(size-1)/2)**2/(2*sigma**2)))
    return gk / np.sum(gk) 
    
    
k_shifts = np.array([-3, -2,-1, 0, 1, 2, 3])    
g = gaussian_kernel(7)     
print(f"g={g}")

def show_image(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)
#Основная функция   
def iris_search(gray, x, y, r, param):
    
    shift = 0
    x, y = x, (y//param) + shift
    center = (x, y)
    r //= param
    if (r < 46/param):
        min_rad = int(r*2.1)
    else:
        min_rad = int(r*1.4) 

    max_rad = int(min(x*0.9, (gray.shape[1]-x)*0.9))
    area = []
    area.append(gray[y-r:y, x-max_rad:x-r])
    area.append(gray[y:y+r, x-max_rad:x-r])
    area.append(gray[y-r:y, x+r+1:x+max_rad])
    area.append(gray[y:y+r, x+r+1:x+max_rad])
    areas_sum = 0
    measure_arc = []
    for i in range (4):
        #show_image(area[i])
        measure_arc.append(90*np.sum(area[i]))
        areas_sum += np.sum(area[i])
    measure_arc /= areas_sum 
    #cv2.rectangle(gray, (x-max_rad, y-min_rad+shift), (x+max_rad,y+min_rad+shift), (0, 255, 0), 2)    
    
    
    
    #print (measure_arc)
    coeff = (measure_arc[3] + measure_arc[2])/(measure_arc[1] + measure_arc[0])

    
    measure_arc = np.radians(measure_arc)
    measure_arc[0] = np.pi - measure_arc[0]
    measure_arc[1] = np.pi + measure_arc[1]
    measure_arc[3] = -measure_arc[3]
    #print (measure_arc)
    max_rad = gray.shape[1]//2
    n_delta_r = np.arange(min_rad, max_rad)
    
    all_rad = n_delta_r[:, None] +  k_shifts[None, :]
    vectorize_integrate = np.vectorize(integral, excluded={0, 1, 2, 3})
    t2 = vectorize_integrate(gray, center, measure_arc[2], measure_arc[3], all_rad) 
    t1 = vectorize_integrate(gray, center, measure_arc[0], measure_arc[1], all_rad) 
    results = (t1 + t2) 
    result_shifts = (results[1:] - results[:-1]) @ g
    """
    results1 = t1 @ g
    results2 = t2 @ g
    result_shifts1 = results1[1:] - results1[:-1]
    result_shifts2 = results2[1:] - results2[:-1]
    result_shifts = resu
    lt_shifts1 + result_shifts2
    """
    """
    max_val1 = np.amax(result_shifts1)
    idx1 = np.argmax(result_shifts1)    
    max_val2 = np.amax(result_shifts2)
    idx2 = np.argmax(result_shifts2)  
    
    max_val = max_val1 + max_val2
    """
    max_val = np.amax(result_shifts)
    idx = np.argmax(result_shifts)    
    
    #result_shifts = result_shifts1 + result_shifts2
    #max_val = np.amax(result_shifts)
    #idx = np.argmax(result_shifts)    
    iris_radius = int(n_delta_r[idx]) * param
    iris_centroid = (int(x*param), int((y-shift)*param))
    return iris_radius, max_val, iris_centroid[0], iris_centroid[1]
    

#-------------------MAIN-------------------#
it = 0
for name in glob.glob("D:/CASIA-IrisV4-Interval/**/*[0-9].jpg", recursive=True):
    iris = cv2.imread(name, 1)
    
    #pupil part
    frame = iris.copy()
    centroid_pupil, radius_pupil = pupil_search(frame)
    print("PUPIL", centroid_pupil, radius_pupil)
    
    #iris part
    gray = iris.copy()
    param = 3
    shift = round(310/radius_pupil)
    x = np.arange(int((centroid_pupil[0] - shift)/param), int((centroid_pupil[0] + shift)/param))
    y = np.arange(int((centroid_pupil[1] + 15)), int((centroid_pupil[1]+20)))
    
    xx, yy = np.meshgrid(x, y)
    xx, yy = np.round(xx).astype(np.uint8), np.round(yy).astype(np.uint8)
   
    gray = image_preprocessing(gray, param)
    vectorize_find_iris = np.vectorize(iris_search, excluded={0, 3})
    all_rad, max_grad, x1, y1 = vectorize_find_iris(gray, xx, yy, radius_pupil, param) 
    all_rad, max_grad, x1, y1 = all_rad.flatten(), max_grad.flatten(), x1.flatten(), y1.flatten()
    idx = np.argmax(max_grad)
    radius_iris = all_rad[idx]
    centroid_iris = (x1[idx], y1[idx] - 10)
    print("IRIS", centroid_iris, radius_iris)
    #writing images
    
    cv2.circle(iris, centroid_pupil, radius_pupil, (169, 245, 9), 2)
    cv2.circle(iris, centroid_iris, int(radius_iris), (87, 4, 89), 2)

    
    #filename = "D:/IRIS/three/"
    filename = 'D:/IRIS/three/image_{}.jpg'.format(it)
    #name = os.path.basename(name)
    #print(filename + name)
    #cv2.imwrite(filename + name, iris)
    cv2.imwrite(filename, iris)
    it+=1
        