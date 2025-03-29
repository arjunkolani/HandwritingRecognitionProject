import cv2
import numpy as np
#import matplotlib.pyplot as plt


def resizeImage(image: np.ndarray) -> np.ndarray:
    """
        Function resizes image to width 1000, maintaining aspect ratio.

        Parameters:
        - image (np.ndarray): The input image to resize.

        Returns:
        - np.ndarray: The resized image.

    """
    height, width, _ = image.shape  # Gets dimensions of image

    new_width = 1000
    new_height = int((new_width / width) * height)  # Maintains original aspect ratio

    new_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return new_image


def thresholdImage(image: np.ndarray) -> np.ndarray:
    """
        Function thresholds image - turns into binary (black and white).

        Parameters:
        - image (np.ndarray): The image to threshold.

        Returns:
        - np.ndarray: The thresholded image.

    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Converts to grayscale image
    # Thresholding turns the image into binary (like black and white)
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    return thresh


def extractLines(thresh_image: np.ndarray) -> list[np.ndarray]:
    """
        Function extracts a list of all the contours of lines from a thresholded image.

        Parameters:
        - thresh_image (np.ndarray): The thresholded image.

        Returns:
        - list: A list of all the contours of lines (contour - NumPy array of (x,y) coordinates of boundary points of
        the line).
    """
    kernel = np.ones((1, 70), np.uint8) # Creates kernel for dilating of (height, width)
    dilated_img = cv2.dilate(thresh_image, kernel, iterations=1) # Creates dilated image using kernel above
    # This dilated image will blur words far more in the x-axis and so will blur words into lines

    # OpenCV function to find outline of the blurred lines
    contours, _ = cv2.findContours(dilated_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Sorts lines on their y co-ordinate (top to bottom)
    sorted_contours_lines = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[1])

    return sorted_contours_lines


def extractWords(image_path: str) -> list[np.ndarray]:
    """
        Function extracts a list of all the sub-images of words in the image specified by the image path given.

        Parameters:
        - image_path (str): The file path of the image to get the words extracted from.

        Returns:
        - list: A list of images (stored as NumPy arrays) of all the individual words in the image of text.
    """
    img = cv2.imread(image_path)  # Reads image from path
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Turns into RGB colour space

    img = resizeImage(img)  # Resize our image
    thresh_img = thresholdImage(img)  # Threshold image to make it easier to use findContours function

    lines = extractLines(thresh_img)  # Gets contours of lines

    # Now we dilate to blur only letters in words together
    kernel = np.ones((15, 12), np.uint8)

    """
        # Code used in iterative development for visualizing the extractLines() function
        img2 = img.copy()  # Storing a copy to draw our rectangles around each word for visualizing function
    
        for line in lines:
            xline, yline, wline, hline = cv2.boundingRect(line)
            cv2.rectangle(img2, (xline, yline), (xline + wline, yline + hline), (0, 255, 0), 2)
    
        plt.imshow(img2)
        plt.show()
    """


    #img2 = img.copy()  # Storing a copy to draw our rectangles around each word for visualizing function
    words = list() # The list of the images of each word

    # We iterate through each line (which are sorted from top to bottom in the extractLines() function)
    # Then for each line we have a nested for loop to iterate through the words (sorted from left to right)
    for line in lines:
        x, y, w, h = cv2.boundingRect(line) # Gets coordinates of line around each rectangle
        line_area = thresh_img[y:y + h, x:x + w] # Gets that particular line part of the thresholded image

        dilated_line = cv2.dilate(line_area, kernel, iterations=1)
        # We find the contours of each word in our image of a dilated line
        contours, _ = cv2.findContours(dilated_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Sorts the contours by their x co-ordinate, left to right
        sorted_contours_words = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[0])


        for word in sorted_contours_words:
            # Selection statement makes sure to only include areas big enough to be a word - so we ignore random small dots
            if cv2.contourArea(word) > 300:
                x2, y2, w2, h2 = cv2.boundingRect(word)

                # We get sub-image of a word
                word_image = img[y + y2: y + y2 + h2, x + x2: x + x2 + w2]


                # Adds rectangular image of each word to words list which is returned by the function
                words.append(word_image)

                # Draws rectangles around words so we can test function during development
                # Will be commented out after function is tested
                #cv2.rectangle(img2, (x + x2, y + y2), (x + x2 + w2, y + y2 + h2), (0, 255, 0), 2)

    #plt.imshow(img2)
    #plt.show()

    return words


def drawWords(image_path: str) -> np.ndarray:
    """
        Function draws rectangles around all the words in the image specified by the image path given.

        Parameters:
        - image_path (str): The file path of the image to draw boxes around words.

        Returns:
        - np.ndarray: The image with rectangle boxes around each word.
    """
    # We use the code from extractWords() function and just keep in our visualizing logic
    img = cv2.imread(image_path)  # Reads image from path
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Turns into RGB colour space

    img = resizeImage(img)  # Resize our image
    thresh_img = thresholdImage(img)  # Threshold image to make it easier to use findContours function

    lines = extractLines(thresh_img)  # Gets contours of lines

    # Now we dilate to blur only letters in words together
    kernel = np.ones((15, 12), np.uint8)

    img2 = img.copy()  # Storing a copy to draw our rectangles around each word for visualizing function
    for line in lines:
        x, y, w, h = cv2.boundingRect(line)
        line_area = thresh_img[y:y + h, x:x + w]

        dilated_line = cv2.dilate(line_area, kernel, iterations=1)
        # We find the contours of each word in our image of a dilated line
        contours, _ = cv2.findContours(dilated_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Sorts the contours by their x co-ordinate, left to right
        sorted_contours_words = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[0])


        for word in sorted_contours_words:
            if cv2.contourArea(word) > 300:
                x2, y2, w2, h2 = cv2.boundingRect(word)

                # Draws rectangles around words
                cv2.rectangle(img2, (x + x2, y + y2), (x + x2 + w2, y + y2 + h2), (0, 255, 0), 2)

    return img2


# Comments below just show some code used for iterative testing and visualisation of the functionality of the above functions
"""
test_path = "data/sample1.png"

# Plots/shows images of the first 10 extracted words
extracted_words = extractWords(test_path)
for extracted_word in extracted_words[:10]:
    plt.imshow(extracted_word)
    plt.show()
"""

"""
# Draws rectangles around each line
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
