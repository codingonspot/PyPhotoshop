from image import Image
import numpy as np

def adjust_brightness(image, factor):
    x_pixels, y_pixels, num_channels = image.array.shape
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)

    # for x in range(x_pixels):
    #     for y in range(y_pixels):
    #         for c in range(num_channels):
    #             new_im.array[x, y, c] = image.array[x, y, c] * factor

    # vectorized version
    new_im.array = image.array * factor

    return new_im

def adjust_contrast(image, factor, mid=0.5):
    x_pixels, y_pixels, num_channels = image.array.shape
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)

    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                new_im.array[x, y, c] = (image.array[x, y, c] - mid) * factor + mid

    # vectorized version
    # new_im.array = (image.array - mid) * factor + mid

    return new_im

def blur(image, kernel_size):
    x_pixels, y_pixels, num_channels = image.array.shape
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)

    neighbour_range = kernel_size // 2

    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                total = 0
                for x_i in range(max(0,x-neighbour_range), min(x_pixels-1, x+neighbour_range) + 1):
                    for y_i in range(max(0, y - neighbour_range), min(y_pixels-1, y+neighbour_range) + 1):
                        total += image.array[x_i, y_i, c]
                new_im.array[x, y, c] = total / (kernel_size ** 2)  #average

    return new_im

def apply_kernel(image, kernel):
    x_pixels, y_pixels, num_channels = image.array.shape
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)

    kernel_size = kernel.shape[0]
    neighbour_range = kernel_size // 2

    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                total = 0
                for x_i in range(max(0,x-neighbour_range), min(x_pixels-1, x+neighbour_range) + 1):
                    for y_i in range(max(0, y - neighbour_range), min(y_pixels-1, y+neighbour_range) + 1):
                        x_k = x_i + neighbour_range - x
                        y_k = y_i + neighbour_range - y
                        kernel_val = kernel[x_k, y_k]
                        total += image.array[x_i, y_i, c] * kernel_val
                    new_im.array[x, y, c] = total

    return new_im


def combine_images(image1, image2):
    x_pixels, y_pixels, num_channels = image1.array.shape
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)

    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                new_im.array[x, y, c] = (image1.array[x, y, c]**2 + image2.array[x, y, c]**2)**0.5

    return new_im


if __name__ == '__main__':
    lake = Image(filename='lake.png')
    city = Image(filename='city.png')

    brigthened_im = adjust_brightness(lake, 1.8)
    brigthened_im.write_image('brightened.png')

    brigthened_im = adjust_brightness(lake, 0.3)
    brigthened_im.write_image('darkened.png')

    incr_contrast = adjust_contrast(lake, 2, 0.5)
    incr_contrast.write_image('incr_contrast.png')

    decr_contrast = adjust_contrast(lake, 0.5, 0.5)
    decr_contrast.write_image('decr_contrast.png')

    blur_3 = blur(city, 3)
    blur_3.write_image('blur_k3.png')

    # let's apply a sobel edge detection kernel on the x and y axis
    sobel_x_kernel = np.array([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]
                               ])
    sobel_y_kernel = np.array([[1, 0, -1],
                               [2, 0, -2],
                               [1, 0, -1]
                               ])

    sobel_x = apply_kernel(city, sobel_x_kernel)
    sobel_x.write_image('edge_x.png')
    sobel_y = apply_kernel(city, sobel_y_kernel)
    sobel_y.write_image('edge_y.png')
    sobel_xy = combine_images(sobel_x, sobel_y)
    sobel_xy.write_image('edge_xy.png')





