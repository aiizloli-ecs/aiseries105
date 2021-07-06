import cv2
import func_q1 as f

alpha = 0
logo_file = 'res-cv-test/samples/blackpink/bp-logo.png'
logo = cv2.imread(logo_file, cv2.IMREAD_UNCHANGED)

image_files = ['jennie.jpg', 'jisoo.jpg', 'lisa.jpg', 'rose.jpg']
titles = ['Jennie', 'Jisoo', 'Lisa', 'Rose']
originals = []
images = []
for i in image_files:
    image = cv2.imread('res-cv-test/samples/blackpink/' + i)
    originals.append(image)
    images.append(image.copy())

# Here, you will modify this program for Question 1.
# The images are already placed inside originals and images[] for you.

# The originals[] is to remain untouched. Your job is to add the Blackpink logo
# (i.e. bp-logo.png) on the eyes of the members in the images inside images[],
# using any of the provided haarcascade models. The routines below will handle
# displaying the images for you.

# To apply the logo to the eyes of the members, use the function:
#     f.apply_logo(image, logo, alpha, x, y, w, h)
# Where:
#     'image': the background image to apply the logo to
#     'logo': the logo to apply (this is already created for you on Line 6)
#     'alpha': the transparency value (this is already created for you on Line 4)
#     'x','y','w','h': the dimension of the eye area on the image to apply the logo to.

titles += titles
originals += images
f.plot_gallery(originals, titles)