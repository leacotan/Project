__author__ = 'LEA'
import Isopod_detection
import skimage
import cv2
import numpy as np
from scipy import misc  as misc
import scipy
from scipy.misc import imread as imread

from skimage import restoration
import matplotlib.pyplot as plt
from skimage import color
import cv2
import matplotlib.image as mpimg
from scipy import ndimage
import matplotlib
import sol1
#

import numpy as np
import matplotlib.pyplot as plt

# from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
# from skimage.filters import roberts, sobel, scharr, prewitt
# from skimage.filters import try_all_threshold
#
# # """background deletion"""
# # image  = color.rgb2gray(imread("hidden_iso.jpg",0)).astype(np.float32)
# # background = color.rgb2gray(imread("background.jpg")).astype(np.float32)
# # forground = np.floor(image*255 - background*255)
# # plt.imshow(forground, cmap = plt.cm.gray)
# # plt.title("background deletion")
# # plt.show()
#
# image = Isopod_detection.read_image("iso.night.jpg",1)
# background = Isopod_detection.create_background(r'D:\LEA\BIOINFORMATICS\Year_3\Project\Image_examples\cage#3-dark\backgrounds')
# plt.title("background ")
# plt.imshow(background,cmap="gray")
# plt.show()
# background, _, __ = sol1.histogram_equalize(background)
# image, _, __ = sol1.histogram_equalize(image)
#
# plt.title("background+ histogram ew ")
# plt.imshow(background,cmap="gray")
# plt.show()
#
#
# forground = np.floor(image*255 - background*255)
# # forground = forground*255
# # forground = (forground + 255)
# forground = 255 + forground
# plt.title("equalized background deletion")
# plt.imshow(forground,cmap="gray")
# plt.show()
#
# blur_radius = 5
# forground = ndimage.gaussian_filter(forground, blur_radius)
#
#
# forground[forground>160] = 255
# forground[forground<160] = 0
# # forground[forground>0] = 255
#
# forground/= 255
# plt.imshow(forground, cmap = plt.cm.gray)
# plt.title("background deletion + blurring + theresholding at 170")
# plt.show()
#
#
#
# """night - background deletion, hist eq , blurring"""
# # blur_radius = 0.6
# #
# # night_back = cv2.imread('background.night.jpg', 0)
# # # equ_back = cv2.equalizeHist(night_back)
# #
# # # equ_back = ndimage.gaussian_filter(equ_back, blur_radius)
# # equ_back = ndimage.gaussian_filter(night_back, blur_radius)
# # plt.imshow(equ_back, cmap = plt.cm.gray)
# # plt.show()
# # night_iso = cv2.imread('iso.night.jpg', 0)
# # # equ_iso = cv2.equalizeHist(night_iso)
# # # equ_iso = ndimage.gaussian_filter(equ_iso, blur_radius)
# # equ_iso = ndimage.gaussian_filter(night_iso, blur_radius)
# #
# # plt.imshow(equ_iso, cmap = plt.cm.gray)
# # plt.show()
# #
# # img =  equ_iso - equ_back
# # plt.imshow(img, cmap = plt.cm.gray)
# # plt.show()
# # plt.imshow(img, cmap = plt.cm.gray)
# # plt.show()
# # equ =  cv2.equalizeHist(img)
# # res = np.hstack((img,equ)) #stacking images side-by-side
# # cv2.imwrite('res_night.png',res)
#
# # a = [(1,2),(3,4)]
# # print(a[:,2])
# """connected components"""
# #
# # # smooth the image (to remove small objects)
# # blur_radius = 1.0
# # imgf = ndimage.gaussian_filter(forground, blur_radius)
# threshold = 1
# # # plt.imshow(imgf, cmap = plt.cm.gray)
# # # plt.show()
# # find connected components
# forground = 1- forground
# plt.imshow(forground, cmap = plt.cm.gray)
# plt.show()
# labeled, nr_objects = ndimage.label(forground > 0)
# # labeled, nr_objects = ndimage.label(imgf )
# print("Number of objects is %d " % nr_objects)
# plt.imsave('connected_comp.png', labeled)
#
# # slices = ndimage.find_objects(labeled)
# # print(slices)
# centers = scipy.ndimage.center_of_mass(forground, labeled)
# print(scipy.ndimage.center_of_mass(forground, labeled, range(1, nr_objects+1)))
# print(len(centers))
#
#
# # matplotlib.axes.Axes.imshow(forground, origin= "upper")
# plt.imshow(forground, cmap = plt.cm.gray)
# plt.show()
# # """smoothing"""
# # kernel = np.ones((3,3),np.float32)/9
# # denoise = cv2.filter2D(forground,-1,kernel)
# # plt.imshow(denoise,cmap=plt.cm.gray)
# # plt.show()
#
# """thresholds"""
# # fig, ax = try_all_threshold(denoise, figsize=(10, 8), verbose=False)
# # plt.show()
# #
# # fig2, ax2 = try_all_threshold(forground, figsize=(10, 8), verbose=False)
# #
# #
# # plt.show()
# # fig1, ax1 = try_all_threshold( image , figsize=(10, 8), verbose=False)
# # plt.show()
#
#
#
#
#
# """sobel"""
# # edge_roberts = roberts(image)
# # edge_sobel = sobel(image)
# #
# # fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True,
# #                        figsize=(8, 4))
# #
# # ax[0].imshow(edge_roberts, cmap=plt.cm.gray)
# # ax[0].set_title('Roberts Edge Detection')
# #
# # ax[1].imshow(edge_sobel, cmap=plt.cm.gray)
# # ax[1].set_title('Sobel Edge Detection')
# #
# # for a in ax:
# #     a.axis('off')
# #
# # plt.tight_layout()
# # plt.show()
#
# '''gamma correction'''
# # image_gamma =imread("test.jpg")
# # gray_gamme = color.rgb2gray(image_gamma)
# # gamma = 2.2
# # print(image_gamma.shape)
# # gray_gamme = gray_gamme**(1/gamma) *255
# # plt.imshow(gray_gamme, cmap = plt.cm.gray)
# # plt.show()
#
# """hist equalization"""
# # img = cv2.imread('test.jpg', 0)
# #
# # equ = cv2.equalizeHist(img)
# # res = np.hstack((img,equ)) #stacking images side-by-side
# # cv2.imwrite('res_BW.png',res)
# #
# #
# #
# # cv2.waitKey()
#
#
# import numpy as np
#
#
# import argparse
# import cv2
#
# #
# #
# def find_circles():
#     # construct the argument parser and parse the arguments
#     ap = argparse.ArgumentParser()
#     ap.add_argument("-i", "--image", required = True, help = "Path to the image")
#     args = vars(ap.parse_args())
#     # load the image, clone it for output, and then convert it to grayscale
#     image = cv2.imread(args["image"])
#     output = image.copy()
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # detect circles in the image
#     circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,1.5, minDist =300, minRadius = 80, maxRadius  =460)
#     # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.8,minDist =300, minRadius = 100, maxRadius  =300)
#
#
#     # ensure at least some circles were found
#     if circles is not None:
#         # convert the (x, y) coordinates and radius of the circles to integers
#         circles = np.round(circles[0, :]).astype("int")
#         dim_x, dim_y = gray.shape[1], gray.shape[0]
#         # loop over the (x, y) coordinates and radius of the circles
#         for (x, y, r) in circles:
#             # draw the circle in the output image, then draw a rectangle
#             # corresponding to the center of the circle
#             if x+r >dim_x or x-r <0 or y+r >dim_y  or y-r <0:
#                 continue
#             cv2.circle(output, (x, y), r, (0, 255, 0), 4)
#             cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
#
#         # show the output image
#         # cv2.imshow("output", np.hstack([image, output]))
#         # cv2.imshow("output", output)
#         cv2.imwrite('output666.jpg',output)
#         cv2.waitKey(0)
#
# find_circles()
# def threshold_image_to_binary(image, threshold):
#     """
#
#     :param image:
#     :param threshold:
#     :return:
#     """
#     image[image > threshold] = 255
#     image[image < threshold] = 0
#     return image
#
# def stretch(image):
#     """
#     stretches an image to [0,1]
#     :param image: the image to stretch
#     :return: the stretched image
#     """
#     minimun, maximum = image.min() , image.max()
#     return (image - minimun) / (maximum - minimun)
#
#
# # image  = color.rgb2gray(imread(r"D:\LEA\BIOINFORMATICS\Year_3\Project\Image_examples\cage#3-light\2015-06-25_05-23-02_775.jpg",0)).astype(np.float32)
# background = color.rgb2gray(imread(r"D:\LEA\BIOINFORMATICS\Year_3\Project\Image_examples\cage#3-light\2015-06-25_09-42-02_150.jpg")).astype(np.float32)
#
# # forground = np.floor(image*255 - background*255)+255
# # #normalize
# forground = stretch(background)
# # # f = np.floor(color.rgb2gray(forground))
# #
#
# # Load picture and detect edges
# image = img_as_ubyte(forground)
# edges = canny(image, sigma=3, low_threshold=5, high_threshold=25)
#
#
# # Detect two radii
# hough_radii = np.arange(100, 150, 5)
# hough_res = hough_circle(edges, hough_radii)
#
# # Select the most prominent 5 circles
# accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
#                                            total_num_peaks=3)
#
# # Draw them
# fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
# image = color.gray2rgb(image)
# for center_y, center_x, radius in zip(cy, cx, radii):
#     circy, circx = circle_perimeter(center_y, center_x, radius)
#     image[circy, circx] = (220, 20, 20)
#
# ax.imshow(image, cmap=plt.cm.gray)
# plt.show()
# # quant = sol1.quantize(forground,7,100 )
# # forground = ndimage.gaussian_filter(quant, 4)
# # forground = threshold_image_to_binary(forground, 0.3)
# # plt.imshow(forground, cmap= "gray")
# # plt.show(
# #
# # night_iso = cv2.imread('test.jpg', 0)
# # equ_iso = cv2.equalizeHist(night_iso)
# # # plt.imshow(equ_iso, cmap = "gray")
# # # plt.show()
# # # denoised = restoration.denoise_bilateral(equ_iso, sigma_color=0.05, sigma_spatial=15, multichannel= False)
# # # denoised = restoration.denoise_nl_means(equ_iso) #bad ouput
# # denoised = restoration.denoise_tv_chambolle(equ_iso)  #good resaults
# # # denoised = restoration.denoise_wavelet(equ_iso) #bad ouput
# # # plt.imshow(denoised, cmap = "gray")
# # # plt.show()
# #
# #
# # res = np.hstack((equ_iso,denoised*255))
# # plt.imshow(res, cmap = "gray")
# plt.show()






import numpy as np
import matplotlib.pyplot as plt

from skimage import color
from skimage.transform import hough_circle
from skimage.feature import peak_local_max
from skimage.draw import circle_perimeter


# # Load picture and detect edges
# image = color.rgb2gray(imread(r"D:\LEA\BIOINFORMATICS\Year_3\Project\Image_examples\cage#3-light\2015-06-25_05-23-02_775.jpg",0)).astype(np.float32)
# edges = skimage.feature.canny(image)
# plt.imshow(edges)
# plt.show()
# fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
#
# # Detect two radii
# hough_radii = np.arange(15, 30, 2)
# hough_res = hough_circle(edges, hough_radii)
#
# centers = []
# accums = []
# radii = []
#
# for radius, h in zip(hough_radii, hough_res):
#     # For each radius, extract two circles
#     peaks = peak_local_max(h, num_peaks=2)
#     centers.extend(peaks - hough_radii.max())
#     accums.extend(h[peaks[:, 0], peaks[:, 1]])
#     radii.extend([radius, radius])
#
# # Draw the most prominent 5 circles
# image = color.gray2rgb(image)
# for idx in np.argsort(accums)[::-1][:5]:
#     center_x, center_y = centers[idx]
#     radius = radii[idx]
#     cx, cy = circle_perimeter(center_y, center_x, radius)
#     image[cy, cx] = (220, 20, 20)
#
# ax.imshow(image, cmap=plt.cm.gray)
# plt.show()

night_image  = color.rgb2gray(imread(r"C:\Users\leact\Project\Predator\7\2015-06-24_19-42-52_228.jpg",0)).astype(np.float32)
day_image  = color.rgb2gray(imread(r"C:\Users\leact\Project\Predator\10\2015-06-25_21-24-38_119.jpg",0)).astype(np.float32)
height, width = day_image.shape
day_image[0:350,650:height] = day_image[0:350,650:height] + 0.2
plt.imshow(day_image)
plt.show()



average_day = np.average(day_image)
average_night= np.average(night_image)
print(str(average_day))
print(str(average_night))
