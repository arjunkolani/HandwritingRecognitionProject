import cv2
import numpy as np
#import matplotlib.pyplot as plt


def resizeImage(image):
    height, width, _ = image.shape # Gets dimensions of image
    new_image = image.copy()
    if width > 1000:
        new_width = 1000
        new_height = int((new_width / width) * height) # Maintains original aspect ratio

        new_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return new_image


def thresholdImage(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Converts to grayscale image
    # Thresholding turns the image into binary (like black and white)
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    return thresh


def extractLines(thresh_image):
    """
        Function takes a thresholded image as a parameter and returns a list of all the contours of lines
    """
    kernel = np.ones((10, 85), np.uint8) # Creates kernel for dilating of height x width = 10 x 85
    dilated_img = cv2.dilate(thresh_image, kernel, iterations=1) # Creates dilated image using kernel above
    # This dilated image will blur words far more in the x-axis and so will blur words into lines

    # OpenCV function to find outline of the blurred lines
    contours, _ = cv2.findContours(dilated_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Sorts lines on their y co-ordinate (top to bottom)
    sorted_contours_lines = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[1])

    return sorted_contours_lines


def extractWords(image_path):
    """
        Function takes the file path of an image and returns a list of images (stored as numpy arrays) of all the words
    """
    img = cv2.imread(image_path)  # Reads image from path
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Turns into RGB colour space

    img = resizeImage(img)  # Resize our image
    thresh_img = thresholdImage(img) # Threshold image to make it easier to use findContours function

    lines = extractLines(thresh_img) # Gets contours of lines

    # Now we dilate to blur only letters in words together
    kernel = np.ones((25, 10), np.uint8)
    dilated = cv2.dilate(thresh_img, kernel, iterations=1)
    # img2 = img.copy()  # Storing a copy to draw our rectangles around each word for visualizing function
    words = list()
    for line in lines:
        x, y, w, h = cv2.boundingRect(line)
        line_area = dilated[y:y + h, x:x + w]
        # We find the contours of each word in our image of a dilated line
        contours, _ = cv2.findContours(line_area.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Sorts the contours by their x co-ordinate, left to right
        sorted_contours_words = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[0])

        for word in sorted_contours_words:
            x2, y2, w2, h2 = cv2.boundingRect(word)
            # Adds rectangular image of each word to words list which is returned by the function
            words.append(img[y+y2: y+y2+h2, x+x2: x+x2+w2])

            # Draws rectangles around words so we can test function during development
            # Will be commented out after function is tested
            # cv2.rectangle(img2, (x + x2, y + y2), (x + x2 + w2, y + y2 + h2), (0, 255, 0), 2)

    """
    plt.imshow(img2)
    plt.show()
    """
    return words


def drawWords(image_path):
    """
        Function takes the file path of an image and returns an image with rectangles around each word.
    """
    img = cv2.imread(image_path)  # Reads image from path
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Turns into RGB colour space

    img = resizeImage(img)  # Resize our image
    thresh_img = thresholdImage(img)  # Threshold image to make it easier to use findContours function

    lines = extractLines(thresh_img)  # Gets contours of lines

    # Now we dilate to blur only letters in words together
    kernel = np.ones((25, 10), np.uint8)
    dilated = cv2.dilate(thresh_img, kernel, iterations=1)
    img2 = img.copy()  # Storing a copy to draw our rectangles around each word for visualizing function
    words = list()
    for line in lines:
        x, y, w, h = cv2.boundingRect(line)
        line_area = dilated[y:y + h, x:x + w]
        # We find the contours of each word in our image of a dilated line
        contours, _ = cv2.findContours(line_area.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Sorts the contours by their x co-ordinate, left to right
        sorted_contours_words = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[0])

        for word in sorted_contours_words:
            x2, y2, w2, h2 = cv2.boundingRect(word)
            # Adds rectangular image of each word to words list which is returned by the function
            words.append(img[y + y2: y + y2 + h2, x + x2: x + x2 + w2])

            # Draws rectangles around words so we can test function during development
            # Will be commented out after function is tested
            cv2.rectangle(img2, (x + x2, y + y2), (x + x2 + w2, y + y2 + h2), (0, 255, 0), 2)

    """
    plt.imshow(img2)
    plt.show()
    """
    return img2


"""
test_path = "data/sample1.png"

extracted_words = extractWords(test_path)
for extracted_word in extracted_words[:10]:
    plt.imshow(extracted_word)
    plt.show()
"""

"""
for line in lines:
    x, y, w, h = cv2.boundingRect(line) # Returns info on rough rectangle around the contour of the line
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 100, 250), 2) # Draws rectangle on image

plt.imshow(img)
plt.show()
"""

"""
# Using matplotlib to visualize our images
rows, columns = 2, 1
fig = plt.figure(figsize=(10, 7))

# Creates first subplot being our original image
fig.add_subplot(rows, columns, 1)
plt.imshow(img)
plt.title("Original")

#Creates second subplot being our resized image
fig.add_subplot(rows, columns, 2)
plt.imshow(resizedImg)
plt.title("Resized")
plt.show()
"""
