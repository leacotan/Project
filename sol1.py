import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.misc import imread as imread
from skimage import color as color
from numpy.linalg import inv

ERR_MSG = "Grayscale to RGB ? you said wou wouldn't...."

INCREMENTN_BY_ONE = 1

FIRST_Z_VAL = 0

GRAY_REPRESENTATION = 1

GRAY_DIM = 2

LAST_PLACE_IN_ARRAY = -1

NORMOLIZED_MAX_VALUE = 1

FIRST_INDEX = 0

THIRD_INDEX = 2

SCONDARY_INDEX = 1

INITIAL_VALUE = 0

MAXIMAL_GRAY_VALUE = 255

ALL_GRAY_VALUES = 256

NUM_CHANELLES =  3

DIM_RGB_IMG = 3

TRANSFORMATION_MATRIX = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])


def transformer(matrix, img):
    """
    preforms a transformation on a given image, using the transformation matrix. used to convert between
    RGB to YIQ
    :param matrix: the matrix to use on the image
    :param img: the image to transform
    :return: the transformed image
    """
    retMatrix = img.copy()
    for i in range(NUM_CHANELLES):
        retMatrix[:,:,i]= matrix[i][INITIAL_VALUE] * img[:, :, INITIAL_VALUE] + matrix[i][SCONDARY_INDEX] * img[:, :, SCONDARY_INDEX] + \
                          matrix[i][THIRD_INDEX] * img[:, :, THIRD_INDEX]
    return retMatrix

def rgb2yiq(imRGB):
    """
    converts an rgb img to yiq, using the conversin matrix
    :param imRGB: the RGB image to convert
    :return: a YIQ image
    """
    return  transformer(TRANSFORMATION_MATRIX, imRGB)

def yiq2rgb(imYIQ):
    """
    converts an yiq img to rgb, using the conversin matrix
    :param imYIQ: the YIQ image to convert
    :return: a RGB   image
    """
    return transformer(inv(TRANSFORMATION_MATRIX), imYIQ)#the matrix nrrds to be inverted

def histogram_equalize(im_orig):
    """
    preforms histogram equalization on the given image, using  a Look Up Table calculated from the formula given
    in the Tirgul. 
    :param im_orig: image to equalize
    :return: the equalized image, the original histogram and the equalized histogram
    """
    #check if the image is RGB, if it is use only the Y Value
    is_RGB = len(im_orig.shape) == DIM_RGB_IMG
    yiq_img = np.array([])
    if is_RGB:
        yiq_img = rgb2yiq(im_orig)
        im_orig = yiq_img[:, :, FIRST_INDEX]

    #turn the image to [0,255]
    im_orig = ((im_orig * MAXIMAL_GRAY_VALUE))

    hist_orig = np.histogram(im_orig, ALL_GRAY_VALUES, [FIRST_INDEX, ALL_GRAY_VALUES])[FIRST_INDEX] #histogram of original
    cum_hist = np.cumsum(hist_orig)
    f = float(im_orig.size)
    cum_hist = cum_hist / f
    max_gray = cum_hist[LAST_PLACE_IN_ARRAY] # topmost, maximal value
    gray_values = np.nonzero(hist_orig)
    max_gray_val = np.amax(gray_values)
    min_gray_val = (gray_values)[FIRST_INDEX][FIRST_INDEX]
    cum_hist  = cum_hist * (max_gray_val)
    #find first nonzero place
    nonzero_index = np.nonzero(cum_hist)
    min_gray = cum_hist[nonzero_index][FIRST_INDEX]
    # calculate LUT according to formula
    LUT = np.round(((cum_hist - min_gray)/(max_gray_val - min_gray ) ) * MAXIMAL_GRAY_VALUE)
    new_im = LUT[(im_orig).astype(np.int64)] #apply LUT to IMG
    hist_equalized = np.histogram(new_im, ALL_GRAY_VALUES, [FIRST_INDEX, ALL_GRAY_VALUES])[0]

    # turn the picture back to [0,1]
    new_im = (new_im / MAXIMAL_GRAY_VALUE).astype(np.float64)

    #if necessary, convert back to RGB
    if is_RGB:
        yiq_img[:, :, FIRST_INDEX] = new_im
        new_im = yiq2rgb(yiq_img)

    #equalized image, 256 bin of original and of equalized
    return new_im.clip(FIRST_INDEX, NORMOLIZED_MAX_VALUE), hist_orig, hist_equalized


def imdisplay(filename, representation):
    """
    open an image with a given filename and representation, grayscale or RGB.
    :param filename: string containing the image filename
    :param representation:either 1 or 2, for grayscale and RGB accordingly
    :return:
    """
    if representation ==GRAY_REPRESENTATION :
        plt.imshow(read_image(filename, representation), cmap=plt.cm.gray)
    else:
        plt.imshow(read_image(filename, representation))


def read_image(filename, representation):
    """
    reads an image and converts it into a given representation- grayscale or RGB.
    :param filename: string containing the image filename
    :param representatnion:  either 1 or 2, for grayscale and RGB accordingly
    :return:
    """

    image = imread(filename)
    matrix = (imread(filename))
    float_matrix = matrix.astype(np.float64)
    is_gray = float_matrix.ndim == GRAY_DIM

    if representation == GRAY_REPRESENTATION:
        if not is_gray:
            float_matrix = color.rgb2gray(image)
            return float_matrix

        float_matrix /= MAXIMAL_GRAY_VALUE
        return float_matrix

    elif representation == GRAY_DIM:
        if is_gray:
            print(ERR_MSG)
            exit()
        else:
            float_matrix /= MAXIMAL_GRAY_VALUE
            return float_matrix

def initializeZ(im_orig, n_quant, hist_orig):
    """
    zet Z (the borders which divide the histogram into segments ) to a division of segmets such that each segment
    will have approx. the same number of pixels (in order to avoid having an empty gray level and a programm crash)
    :param im_orig: the original image to quantize (RGB or grayscale)
    :param n_quant: the number of intensities the quantized image  should have
    :param hist_orig: the original histogram of the image to quantize
    :return:
    """
    cum_hist = np.cumsum(hist_orig)
    z_initial = np.empty(n_quant + INCREMENTN_BY_ONE).astype(np.int64)
    z_initial[FIRST_INDEX] = FIRST_Z_VAL
    num_pixels = (im_orig.size)
    index = 1
    while index <= n_quant:
        z_initial[index] = np.argmax(cum_hist >= (index * (num_pixels/n_quant)))
        index += INCREMENTN_BY_ONE

    z_initial[LAST_PLACE_IN_ARRAY] = MAXIMAL_GRAY_VALUE
    return z_initial

def calculteQ(z_arr, n_quant, hist_img):
    """
    computing the values to which each of the segments' intensities will map.
    :param z_arr: an array size [n_quant +1] , with its first and last entries initialized to 0, 255 accordingly,
    with each entry i marking the end of the i'th segment [exept i = 0]
    :param n_quant: the number of intensities the quantized image  should have
    :param hist_orig: the original histogram of the image to quantize
    :return: an array size [n_quant]  containing the values to which each of the segments' intensities will map.
     """
    q_arr =np.empty(n_quant).astype(np.float64)
    for i in range(n_quant):
        z_i = z_arr[i]
        z_i_plus_one = z_arr[i+INCREMENTN_BY_ONE]
        relevent_hist_piece =hist_img[z_i :z_i_plus_one]
        relevent_arr = np.arange(z_i, z_i_plus_one)
        q_arr[i] = (float(np.sum(relevent_arr *relevent_hist_piece)) / float(np.sum(relevent_hist_piece)))
    return q_arr

def calculateZ(q_arr, n_quant):
    """
    calculating the boarders which divide the histogram into segments.
    :param q_arr: an array size [n_quant]  containing the values to which each of the segments' intensities will map.
    :param n_quant: the number of intensities the quantized image  should have
    :return: an array size [n_quant +1] , with its first and last entries initialized to 0, 255 accordingly,
    with each entry i marking the end of the i'th segment [exept i = 0]
    """
    z_arr= np.empty(n_quant+1).astype(np.int64)
    z_arr[FIRST_INDEX] = FIRST_Z_VAL
    z_arr[n_quant ]= MAXIMAL_GRAY_VALUE
    for i in range( n_quant -1) :
        z_arr[i+1] = (q_arr[i] + q_arr[i+1])/ 2
    return z_arr

def calculateERR(q_arr, z_arr, n_quant, hist_orig):
    """
    calculates an array with [n_iter] (or less) entries of the total intensities error for each iteration of the
    quantization process.
    :param q_arr: : an array size [n_quant]  containing the values to which each of the segments' intensities will map.
    :param z_arr: an array size [n_quant +1] , with its first and last entries initialized to 0, 255 accordingly,
    with each entry i marking the end of the i'th segment [exept i = 0]
    :param n_quant: the number of intensities the quantized image  should have
    :param hist_orig: the original histogram of the image to quantize
    :return: an array with [n_iter] (or less) entries of the total intensities error for each iteration of
    the quantization process.
    """
    error = 0
    for i in range(n_quant):
        z_i = z_arr[i]
        z_i_plus_one = z_arr[i + 1]
        p_z = hist_orig[z_i:z_i_plus_one]
        relevent_arr = np.arange(z_i, z_i_plus_one)
        power = pow((q_arr[i] - relevent_arr), 2)
        power*= p_z
        sum =np.sum(power)
        error += sum
    return error


def quantize(im_orig, n_quant, n_iter):
    """
    preforms optimal quantization of a given RGB or grayscal image
    :param im_orig: the original image to quantize (RGB or grayscale)
    :param n_quant: the number of intensities the quantized image  should have
    :param n_iter: number of iterations to preform.
    :return: the quantized image and an array with [n_iter] (or less) entries of the total intensities error for
    each iteration of the quantization process.
    """
    #check if the image is RGB, if it is use only the Y Value

    is_RGB = len(im_orig.shape) == DIM_RGB_IMG
    yiq_img = np.array([])
    if is_RGB:
        yiq_img = rgb2yiq(im_orig)
        im_orig = yiq_img[:, :, 0]
    # turn the image to [0,255]
    im_orig = np.around((im_orig *MAXIMAL_GRAY_VALUE )).astype( np.int64)
    hist_orig = np.histogram(im_orig, ALL_GRAY_VALUES, [0, ALL_GRAY_VALUES])[0]  # histogram of original
    # initialize z
    z_arr = initializeZ(im_orig,n_quant, hist_orig)
    # initialize q with the initail z
    q_arr = calculteQ(z_arr, n_quant, hist_orig)
    err_arr = np.empty(n_iter).astype(np.float64)
    #initial error
    # err_arr[0] = calculateERR(q_arr, z_arr, n_quant, hist_orig)

    #preform an iteration, updating z, q and err
    for i in range(n_iter):
        updated_z = calculateZ(q_arr, n_quant)
        #check wether the process converged. if so, stop
        if np.array_equal(updated_z, z_arr) :
            break

        q_arr = calculteQ(updated_z, n_quant, hist_orig)
        z_arr = updated_z
        err_arr[i] = calculateERR(q_arr, z_arr,n_quant,hist_orig)
    lookup_table = np.empty([ALL_GRAY_VALUES])
    #build LUT
    for  k in range(n_quant):
        lookup_table[z_arr[k] :z_arr[k+1]] = q_arr[k]
    # apply LUT to IMG
    new_im =lookup_table[(im_orig).astype(np.int64)]

    new_im = new_im/ MAXIMAL_GRAY_VALUE
    #if necessary, convert back to RGB
    if is_RGB:
        yiq_img[:,:,0] = new_im
        new_im = yiq2rgb(yiq_img)

    return (new_im).astype(np.float64)



