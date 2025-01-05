import os
import math

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.layers import StringLookup
from tensorflow import keras

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import language_tool_python


def resizeImage(image):
    height, width, _ = image.shape  # Gets dimensions of image
    """
    new_image = image.copy()
    
    if width > 1000:
        new_width = 1000
        new_height = int((new_width / width) * height) # Maintains original aspect ratio

        new_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    """
    new_width = 1000
    new_height = int((new_width / width) * height)  # Maintains original aspect ratio

    new_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return new_image


def thresholdImage(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Converts to grayscale image
    # Thresholding turns the image into binary (like black and white)
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    return thresh


def extractLines(thresh_image):
    """
        Function takes a thresholded image as a parameter and returns a list of all the contours of lines
    """
    kernel = np.ones((1, 70), np.uint8)  # Creates kernel for dilating of height x width = 3 x 85
    dilated_img = cv2.dilate(thresh_image, kernel, iterations=1)  # Creates dilated image using kernel above
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
    thresh_img = thresholdImage(img)  # Threshold image to make it easier to use findContours function

    lines = extractLines(thresh_img)  # Gets contours of lines

    # Now we dilate to blur only letters in words together
    kernel = np.ones((15, 15), np.uint8)
    """
    img2 = img.copy()  # Storing a copy to draw our rectangles around each word for visualizing function

    
    for line in lines:
        xline, yline, wline, hline = cv2.boundingRect(line)
        cv2.rectangle(img2, (xline, yline), (xline + wline, yline + hline), (0, 255, 0), 2)

    plt.imshow(img2)
    plt.show()
    """

    img2 = img.copy()  # Storing a copy to draw our rectangles around each word for visualizing function
    words = list()
    for line in lines:
        x, y, w, h = cv2.boundingRect(line)
        line_area = thresh_img[y:y + h, x:x + w]

        # Here we dilate on the lines using a now more vertical kernel
        dilated_line = cv2.dilate(line_area, kernel, iterations=1)

        # We find the contours of each word in our image of a dilated line
        contours, _ = cv2.findContours(dilated_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Sorts the contours by their x co-ordinate, left to right
        sorted_contours_words = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[0])

        for word in sorted_contours_words:

            if cv2.contourArea(word) > 300:
                x2, y2, w2, h2 = cv2.boundingRect(word)

                word_image = img[y + y2: y + y2 + h2, x + x2: x + x2 + w2]

                # Adds rectangular image of each word to words list which is returned by the function
                words.append(word_image)

                # Draws rectangles around words so we can test function during development
                # Will be commented out after function is tested
                cv2.rectangle(img2, (x + x2, y + y2), (x + x2 + w2, y + y2 + h2), (0, 255, 0), 2)


    plt.imshow(img2)
    plt.show()

    return words


def preprocessImage(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # increase contrast
    pxmin = np.min(img_gray)
    pxmax = np.max(img_gray)
    imgContrast = (img_gray - pxmin) / (pxmax - pxmin) * 255
    # increase line width
    kernel1 = np.ones((3, 3), np.uint8)
    imgMorph = cv2.erode(imgContrast, kernel1, iterations=1)

    tensor_image = tf.convert_to_tensor(imgMorph, dtype=tf.uint8)
    tensor_shape = tf.shape(tensor_image).numpy()
    tensor_image = tf.reshape(tensor_image, [tensor_shape[0], tensor_shape[1], 1])
    tensor_image = tf.image.resize_with_pad(tensor_image, 32, 128)
    tensor_image = tf.transpose(tensor_image, perm=[1, 0, 2])
    tensor_image = tf.image.flip_left_right(tensor_image)

    tensor_image = tf.cast(tensor_image, tf.float32) / 255.0
    return tensor_image


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
              :, :21
              ]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char2(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


# 100epoch one
num_to_char2 = StringLookup(
    vocabulary=['[UNK]', 's', '3', 'Y', 'G', 'j', 'O', '6', ')', 'p', '-', 'I', 'F', 'P', 'd', 'Q', '!', 'M', 'o', 'T',
                '+', '(', '4', '&', 'f', 'h', '1', 'e', 'K', '#', 'z', 'X', 'c', 'R', 'E', 'L', '9', '2', 'i', '7', '*',
                'B', '/', 'Z', 'W', '8', ';', 'r', 'N', "'", 'U', 'q', ',', 'a', '5', 'J', 'n', 'H', ':', 'C', '"', 'k',
                '.', 'v', 'S', 'V', 'y', 'l', 'x', 'D', '?', 'g', 'w', 'u', '0', 'b', 'A', 'm', 't'],
    mask_token=None, invert=True
)

test_images = ["sample1.png", "sample2.JPEG", "sample3.png", "sample4.jpg"]
for test_image in test_images:
    test_path = "data/" + test_image
    extractWords(test_path)


"""
tensor_images = list()
for extracted_word in extracted_words:
    tensor_images.append(preprocessImage(extracted_word))

tensor_images = tf.convert_to_tensor(tensor_images, dtype=tf.float32)

model_path = 'new_htr_model_100epochs.keras'
new_model = tf.keras.models.load_model(model_path)
print("Hi")
preds = new_model.predict(tensor_images)
pred_texts = decode_batch_predictions(preds)
print(pred_texts)

word_count = len(pred_texts)
size = math.ceil(math.sqrt(word_count))
_, ax = plt.subplots(size, size, figsize=(30, 14))
for i in range(size ** 2):
    if (i < word_count):
        img = tensor_images[i]
        img = tf.image.flip_left_right(img)
        img = tf.transpose(img, perm=[1, 0, 2])
        img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
        img = img[:, :, 0]

        title = f"Prediction: {pred_texts[i]}"
        ax[i // size, i % size].imshow(img, cmap="gray")
        ax[i // size, i % size].set_title(title)
        ax[i // size, i % size].axis("off")

plt.show()

output = " ".join(pred_texts)
print(output)
"""
"""
tool = language_tool_python.LanguageToolPublicAPI('en-US')

for i in range(10):
    print(tool.check(output))

"""