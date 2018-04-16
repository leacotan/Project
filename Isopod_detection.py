__author__ = 'LEA'
from scipy import ndimage
import scipy
import sys
import os
import sol1
from skimage import restoration
from skimage.morphology import ball

import cv2
import FileWriter

from matplotlib.patches import Circle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.misc import imread , imsave
from skimage import color as color
from skimage import morphology as morph
from PIL import Image
from pathlib import Path
import time


light_or_dark = ""

def create_circled_background(circles, image, background_dir_name):
    for x,y,r in circles:
        circled_image = draw_circle_outline(image, x, y,r )

    imsave(os.path.join(background_dir_name,("background_with_circles.png")), circled_image)

def draw_circle_outline(image, x_center, y_center, radius, value=0):
    """
    draws a circle in given coordinates.
    :param image:
    :param x_center:
    :param y_center:
    :param radius:
    :param value:
    :return:
    """
    new_image = np.copy(image)
    (height, width) = image.shape
    for x in range(width):
        for y in range(height):
            if (x - x_center)**2 + (y-y_center)**2 == radius**2:
                new_image[y][x] = value
    return new_image

def draw_circle(image, x_center, y_center, radius, value=0):
    """
    draws a circle in given coordinates.
    :param image:
    :param x_center:
    :param y_center:
    :param radius:
    :param value:
    :return:
    """
    new_image = np.copy(image)
    (height, width) = image.shape
    for x in range(width):
        for y in range(height):
            if (x - x_center)**2 + (y-y_center)**2 <= radius**2:
                new_image[y][x] = value
    return new_image

def tag_circles(circles, directory):
    if circles is None:
        print("No circles found. Check manually in directory :\n" + directory)
        return (0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0)
    is_in_left_side = lambda left:left <= 400
    is_in_right_side = lambda right:right >= 600
    is_in_upper_side = lambda up:up <= 500
    is_in_lower_side = lambda down:down >= 800
    scorpion, earth, leaf, black, white = (0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0)
    hight_smaller_then_upper = 500
    hight_bigger_then_lower = 800
    width_smaller_then_left = 400
    width_bigger_then_right = 600
    for circle in circles:
        x,y, r = circle
        if is_in_left_side(x):
            if is_in_upper_side(y):
                white = circle
            elif is_in_lower_side(y):
                scorpion = circle
        elif is_in_right_side(x):
            if is_in_upper_side(y):
                earth = circle
            elif is_in_lower_side(y):
                 leaf = circle
        else:
            black = circle
    if scorpion == (0,0,0) :
        print("Scorpion not found. Check manually in directory :\n" + directory)
    if earth == (0, 0, 0):
        print("Earth dish not found. Check manually in directory :\n" + directory)
    if leaf == (0, 0, 0):
        print("Leaf dish not found. Check manually in directory :\n" + directory)
    #todo watch out, this logic may not be right for all the pictures
    return scorpion, earth, leaf, black, white




def show_image(image):
    """
    a helper function used to show images in one line
    :param image:
    :return:
    """
    plt.imshow(image, cmap = "gray")
    plt.show()


def find_circles(input_image_filename ):
    """
    uses hough transform to find circles in the image.
    :param input_image_filename:
    :return:
    """

    # load the image, clone it for output, and then convert it to grayscale
    image = cv2.imread(input_image_filename)

    output = image.copy()
    # gray = cv2.imshow(image, gray)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.8,minDist =300, minRadius = 100, maxRadius  =300)

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        dim_x, dim_y = gray.shape[1], gray.shape[0]
        updated_circles = []
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            if x+r >dim_x or x-r <0 or y+r >dim_y  or y-r <0:
                continue
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            updated_circles.append((x,y,r))
        # show the output image
        # cv2.imshow("output", np.hstack([image, output]))
        # cv2.waitKey(0)
        return np.array(updated_circles)


def read_image(filename, representation):
    """
    reads an image and converts it into a given representation- grayscale or RGB.
    :param filename: string containing the image filename
    :param representatnion: either 1 or 2, for grayscale and RGB accordingly
    :return: the converted image
     """
    image = imread(filename)
    float_matrix = (imread(filename)).astype(np.float64)
    is_gray = float_matrix.ndim == 2
    if representation == 1:
        if not is_gray:
            float_matrix = color.rgb2gray(image)
            return float_matrix

        float_matrix /= 255
        return float_matrix
    elif representation == 2:
            float_matrix /= 255
    return float_matrix



def create_background(background_dir_name):
    """
    creates a background image that is the avarage of all the images in its directory
    :param background_dir_name: the directory of images from which to create the image
    :return: the background image
    """
    # Access all  JPG files in directory
    allfiles = os.listdir(background_dir_name)
    imlist= []
    for filename in allfiles:
        if filename[-4:] in [".jpg"] and not filename.startswith("background")and not filename.startswith("detected"):
            imlist.append(os.path.join(background_dir_name, filename))

    # Assuming all images are the same size, get dimensions of first image
    #read the image as grayscale
    image = read_image(imlist[0],1)

    N = len(imlist)

    # Create a numpy array of floats to store the average (assume RGB images)
    arr = np.zeros(image.shape,np.float)

    # Build up average pixel intensities, casting each image as an array of floats
    for im in imlist:
        imarr = read_image(im,1)
        arr = arr + imarr/N

    # Round values in array and cast as 8-bit integer
    # arr = np.array(np.round(arr),dtype=np.uint8)


    # out = plt.imshow(arr, cmap = "gray")
    # Generate, save and preview final image
    # plt.axis('off')
    # plt.savefig(os.path.join(background_dir_name,("background.jpg")),  bbox_inches='tight')
    # # plt.show()
    #
    imsave(os.path.join(background_dir_name,("background.png")), arr)
    global light_or_dark
    if np.average(arr)> 0.3:
       light_or_dark = "light"
    else:
        light_or_dark = "dark"

    return arr

def stretch(image):
    """
    stretches an image to [0,1]
    :param image: the image to stretch
    :return: the stretched image
    """
    minimun, maximum = image.min() , image.max()
    return (image - minimun) / (maximum - minimun)

def threshold_image_to_binary(image, threshold):
    """

    :param image: :  the name of the image
    :param threshold: if a pixel value is bigger hen threshold (lighter)- turn it white
    else: turn in black
    :return: the binary image
    """
    image[image > threshold] = 255
    image[image < threshold] = 0
    return image



def find_izo_night(image, background, saf_thersh = 120, blur_radius=3):
    """
    :param image:  the name of the image
    :param background: the background image of this group of photos
    :return: coordinates, number_objects
    """
    forground = np.floor(image*255 - background*255)
    # forground = stretch(forground )*255

    forground = (forground + 255)


    # forground = stretch(forground )*255
    forground = ndimage.gaussian_filter(forground, blur_radius)
    # binary_iso = threshold_image_to_binary(forground, saf_thersh)
    binary_iso = threshold_image_to_binary(forground, saf_thersh) / 255
    threshold = 1
    forground = 1 - forground
    labeled, nr_objects = ndimage.label(binary_iso < threshold)
    coordinates = scipy.ndimage.center_of_mass(1-binary_iso, labeled, range(1, nr_objects +1))
    bincount = (np.bincount(labeled.flatten()))[1:,]
    if nr_objects != 0:
       calculate_avg_iso_size(bincount)
    return coordinates, nr_objects

def find_izo_day(image, background, saf_thersh,scorpion_circle, blur_radius=3):
    """
    :param image:  the name of the image
    :param background: the background image of this group of photos
    :return: coordinates, number_objects
    """
    forground = np.abs(image - background)

    forground = np.floor(forground*255)

    # plt.hist(forground.flatten(), bins = list(range(256)), log = True)
    # plt.show()

    #get rid of scorpion
    forground = draw_circle(forground,*scorpion_circle)
    #blur
    forground = ndimage.gaussian_filter(forground, blur_radius)


    image_histogram, bins = np.histogram(forground.flatten(), 256, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = cdf / cdf[-1]
    # plt.plot(cdf)
    # plt.show()

    (height, width) = forground.shape
    for x in range(width):
        for y in range(height):
            if (x >= 650) and (y<=350):
                if cdf[int(forground[y][x])] < 0.97:
                    forground[y][x] = 0
            elif cdf[int(forground[y][x])] < 0.99:
                forground[y][x] = 0
            # elif cdf[int(forground[y][x])] < 0.97:
            #     forground[y][x] = 125
            # else:
            #    forground[y][x] = 255

    #seperate using morphology

    # b = ball(3, np.uint8)
    # eroded_forground= morph.erosion(forground)
    # open_forground  = morph.opening(forground, selem  = b)
    labeled, nr_objects = ndimage.label(forground)
    #print(nr_objects)
    CCs = []
    for label in range(1, nr_objects+1):
        Y,X = np.where(labeled == label)
        size = len(X)
        avg_color = 0
        std_color = 0
        avg_x = 0
        avg_y = 0
        for i in range(size):
            x, y = X[i], Y[i]
            avg_color += forground[y][x]
            std_color += forground[y][x]**2
            avg_x += x
            avg_y += y
        avg_color /= size
        std_color /= size
        std_color = (std_color - avg_color**2)**0.5
        avg_x /= size
        avg_y /= size
        CCs.append((label, size, (avg_x, avg_y), avg_color, std_color))
        if size > 100:
            # print(CCs[-1])
            pass

    # show_image(forground)

    object_slices = ndimage.find_objects(labeled)   #a[loc]  A list of tuples, with each tuple containing N slices slices correspond to the minimal parallelepiped that contains the object.

    # forground = stretch(forground )
    # TODO histogram   SILANCE I KILL YOU
    # plt.hist(forground.flatten(), bins = list(range(256)), log = True)
    # plt.show()

    # binary_iso = threshold_image_to_binary(forground, saf_thersh) / 255
    # threshold = 1
    # forground = 1 - forground
    # labeled, nr_objects = ndimage.label(forground < threshold)


    size_threshold = 200
    coordinates = np.array(scipy.ndimage.center_of_mass(1-forground, labeled, range(1, nr_objects +1)))
    bincount = (np.bincount(labeled.flatten()))[1:,]

    #filter those with size smaller then 20 pixels
    coordinates = coordinates[np.where(bincount>size_threshold)]
    nr_objects = len(coordinates)
    if nr_objects > 16:
        print("Found more than 16 isopods. Taking higher threshold of size")
        coordinates = coordinates[np.where(bincount > size_threshold +50)]
        nr_objects = len(coordinates)

    if nr_objects != 0:
       calculate_avg_iso_size(bincount[bincount>size_threshold])
    return coordinates, nr_objects



def mark_red(coordinates, nr_object):
    """
    :param coordinates: tuples of ((x1,y1)...(xn,yn))
    :param nr_object: how many object were found in the picture
    :return:  list of (x1, x2...xn) and (y1....yn)
    """
    x= []
    y = []
    for i in range(nr_object):
        y.append(coordinates[i][0])
        x.append(coordinates[i][1])
    return x,y


def check_iso_loc(directory, pic_name, background, saf_thresh, blur_radius , scorpion_loc_tuple):
    """
    finds the isopods in the image, then outputs into a file.
    :param directory: the directory in which  to save the image
    :param pic_name: the name of the image
    :param background: the background image of this group of photos
    :param saf_thresh: the threshold by which to filter isopods
    :param blur_radius: how much to blur
    :param light_or_dark: are these day or night pictures
    :param circles: the circles found in the background picture
    :return:
    """
    image = read_image(os.path.join(directory,pic_name) , 1 )
    use_circles = True
    global light_or_dark
    if light_or_dark == "dark":

        image ,_,__ = sol1.histogram_equalize(image)
        # image = restoration.denoise_tv_chambolle(image)
        background, _, __ = sol1.histogram_equalize(background)
        # background = restoration.denoise_tv_chambolle(background)
        coor, num_iso = find_izo_night(image, background, saf_thresh, blur_radius)

        use_circles = False
    else:
        # background = read_image(os.path.join(directory,background_name) , 1)
        coor, num_iso = find_izo_day(image, background, saf_thresh, scorpion_loc_tuple, blur_radius)


    legal_cur  = coor


    if True: #todo change this when in need of mrking
        x,y = mark_red(coor, num_iso)
        implot = plt.imshow(image, cmap = "gray")

        # put a red dot, size 10, at x,y locations:
        plt.scatter(x,y, c='r', s=10)
        plt.axis('off')
        plt.savefig(os.path.join(directory,("detected_blur_radius_"+ str(blur_radius)+"thresh"+str(saf_thresh)+ str(pic_name))),  bbox_inches='tight')
        # plt.savefig(os.path.join(directory,("detected_blur_radius_"+ str(blur_radius)+"thresh"+str(0.5)+ str(pic_name))))

        #clears the figure so that they don't accumulate
        plt.clf()
        plt.cla()
        plt.close()
    return legal_cur, num_iso

avg_size_iso = 0
min_size_iso = 0
max_size_iso = 0
iso_sizes = []



def main(directory,dest_directory,threshold, blur_radius ):

    # print("Usage: directory(path) threshold(int, ~160) blur_radius(~4) light/dark(string) )")

    global avg_size_iso
    global min_size_iso
    global max_size_iso

    background_dir_name = directory
    #checks if there is a need to calculate the background
    backgournd_file = Path(os.path.join(background_dir_name,("background.png")))
    background = None
    if not backgournd_file.is_file():
        background = create_background(background_dir_name) #black and white background
    else:
        background  = sol1.read_image(os.path.join(background_dir_name,("background.png")),1)

    cur_file_writer = FileWriter.FileWriter(dest_directory,directory)
    #find the circles- a list of (x, y, r ) with x y as center of circles
    background_image_name = os.path.join(background_dir_name,("background.png"))
    if Path(os.path.join(background_dir_name,("background_with_circles.png"))).is_file():
        background_image_name = os.path.join(background_dir_name,("background_with_circles.png"))
    circles = find_circles(background_image_name)
    tagged_circles = tag_circles(circles, directory)
    create_circled_background(circles, background, background_dir_name)
    cur_file_writer.write_background(*tagged_circles)


    scorpion_x, scorpion_y, scorpion_r = tagged_circles[0]
    # radiuses = circles[:,2]



    for filename in os.listdir(directory):
         if filename.endswith(".jpg") and not filename.startswith("detected") and not filename.startswith("background"):
          coordinates, num_iso = check_iso_loc( directory, filename ,background, threshold,blur_radius, tagged_circles[0])
          cur_file_writer.write_image_data(filename, coordinates=coordinates, num_isopods= num_iso)
    print("avg iso size is : "+ str(avg_size_iso))
    print("min iso size is : "+ str(min_size_iso))
    print("max iso size is : "+ str(max_size_iso))
    flat_list = [item for sublist in iso_sizes for item in sublist]
    plt.hist(flat_list,bins = list(range(min_size_iso, max_size_iso+10, 10)), range = (np.min(flat_list), np.max(flat_list)))
    # plt.show()

def calculate_avg_iso_size(bincount):
    """
    calculates the avg, min amd max size of the isopods
    :param bincount: how many isopods are there
    :return:
    """
    global avg_size_iso
    global min_size_iso
    global max_size_iso
    global iso_sizes
    nonzero_bins = bincount
    cur_avg = np.average(nonzero_bins)
    cur_min_izo = np.min(nonzero_bins)
    cur_max_izo = np.max(nonzero_bins)

    iso_sizes.append(bincount.flatten())

    if min_size_iso == 0:
        min_size_iso = cur_min_izo
    else:
        min_size_iso = min(min_size_iso, cur_min_izo)

    if avg_size_iso == 0 :
       avg_size_iso = cur_avg
    else:
        avg_size_iso = (avg_size_iso + cur_avg)/2
    max_size_iso = max(max_size_iso, cur_max_izo)
