import cv2
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing.pool import ThreadPool
import os
import pandas as pd
import numpy as np
import shutil
from tqdm.notebook import trange, tqdm
from PIL import Image, ImageEnhance, ImageChops
import random

# 필요없는 검정색 배경 부분 삭제 함수
def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        # mask = img>tol

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            # print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img


def circle_crop(img, sigmaX = 30):   
    """
    Create circular crop around image centre    
    """    
    img = crop_image_from_gray(img)    
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img) # 원형으로 사진 cutting 해주는 함수
    img = crop_image_from_gray(img)
    cv2.resize(img, (512, 512))
    return img 

def preprocess_image(file):
    input_filepath ='/Users/ijuyeong/Downloads/intern/DR/aptos2019-blindness-detection/ipynb/data/4/'+file
    output_filepath = '/Users/ijuyeong/Downloads/intern/DR/aptos2019-blindness-detection/ipynb/preprocess/4/'+file
    
    img = cv2.imread(input_filepath)
    img = circle_crop(img) 
    
    print(img.shape)
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    plt.imshow(img)
    cv2.imwrite(output_filepath, cv2.resize(img, (IMG_SIZE,IMG_SIZE)))

'''This Function uses Multi processing for faster saving of images into folder'''

def multiprocess_image_processor(process:int, imgs:list):
    """
    Inputs:
        process: (int) number of process to run
        imgs:(list) list of images
    """
    print(f'MESSAGE: Running {process} process')
    results = ThreadPool(process).map(preprocess_image, imgs)
    return results


IMG_SIZE = 512

folder_path = "/Users/ijuyeong/Downloads/intern/DR/aptos2019-blindness-detection/ipynb/data/4/"
file_list = os.listdir(folder_path)
print(file_list)
if file_list != ".DS_Store":
    file_array = np.array(file_list)
    print(file_array,type(file_array))
    multiprocess_image_processor(6, file_array)