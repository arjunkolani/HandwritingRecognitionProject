import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# This sets an environment variable to stop certain errors occurring

from tensorflow.keras.layers import StringLookup
from tensorflow import keras

import cv2
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf

from wordSegmentation import extractWords
from autocorrect import Speller


def preprocessImage(image: np.ndarray) -> tf.Tensor:
    """
        Function processes image by increasing line contrast and width and transforming into the tensor format required
        for model prediction.

        Parameters:
        - image (np.ndarray): The image to be processed and converted to a tensor.

        Returns:
        - (tf.Tensor): A tensor of the processed image. Also includes transformations made for the images used in the
        training of the model.
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # This grayscales image

    # increases contrast
    pxmin = np.min(img_gray)
    pxmax = np.max(img_gray)
    imgContrast = (img_gray - pxmin) / (pxmax - pxmin) * 255

    # increases line width
    kernel1 = np.ones((3, 3), np.uint8)
    imgMorph = cv2.erode(imgContrast, kernel1, iterations=1)


    tensor_image = tf.convert_to_tensor(imgMorph, dtype=tf.uint8) # Converts numpy array to tensor

    # This adds an extra dimension to match shape of the tensors in the datasets used in training
    tensor_shape = tf.shape(tensor_image).numpy()
    tensor_image = tf.reshape(tensor_image, [tensor_shape[0], tensor_shape[1], 1])

    # These image transformations are the same as for the images in the dataset for training of the model.
    # This is to prepare image for being passed into NN model
    tensor_image = tf.image.resize_with_pad(tensor_image, 32, 128)
    tensor_image = tf.transpose(tensor_image, perm=[1, 0, 2])
    tensor_image = tf.image.flip_left_right(tensor_image)
    tensor_image = tf.cast(tensor_image, tf.float32) / 255.0

    return tensor_image


def decode_predictions(pred: np.ndarray) -> list[str]:
    """
        Function decodes numerical predictions from model into text using CTC decoding and our num_to_char
        mapping defined from the training of the model.

        Parameters:
        - pred (np.ndarray): A NumPy array of each prediction stored in a numerical vector format.

        Returns:
        - (list[str]): A list of strings which are the decoded predictions of each word.

    """
    input_len = np.ones(pred.shape[0]) * pred.shape[1] # Gets dimensions of pred array for decode function below
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :21
    ]
    # Number 21 comes from the fact that the maximum length of each label during training was 21

    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1))) # Removes -1s
        res = tf.strings.reduce_join(num_to_char2(res)).numpy().decode("utf-8")
        # Then converts indices to characters using num_to_char mapping
        output_text.append(res) # Adds each string to list to return
    return output_text


def convert_text(image_path, model_path) -> str:
    """
        Function extracts the text from an image of text using a given neural network model.

        Parameters:
        - image_path (str): File path of image to extract text from.
        - model_path (str): File path of neural network model to use to predict text in image.

        Returns:
        - (str): A string of the text extracted from the image.
    """
    # Gets images of the words from the image using our word segmentation algorithm
    extracted_words = extractWords(image_path)

    # Now we create a list of all our images after preprocessing them and then make this list a tensor
    tensor_images = list()

    for extracted_word in extracted_words:
        tensor_images.append(preprocessImage(extracted_word))

    tensor_images = tf.convert_to_tensor(tensor_images, dtype=tf.float32)

    # This loads the model and predicts the text of all our word images
    pred_model = tf.keras.models.load_model(model_path)
    preds = pred_model.predict(tensor_images)
    pred_texts = decode_predictions(preds)

    output = " ".join(pred_texts)  # Takes our list of predicted words and combines them with spaces in between

    return output


def spell_correct(text: str) -> str:
    """
        Function spell-corrects a piece of text.

        Parameters:
        - text (str): The text to be spell corrected.

        Returns:
        - (str): The corrected text.
    """
    spell = Speller(lang="en")  # Instantiates our spell-checker object
    return spell(text)


# Each model had its own num_to_char mapping from training (seeded randomly in training)
# This is the mapping for the 30 epoch model
num_to_char = StringLookup(
    vocabulary=['[UNK]',
 'S',
 'd',
 '#',
 '"',
 'z',
 '4',
 'B',
 'a',
 'O',
 '5',
 'm',
 'f',
 '1',
 'e',
 'M',
 'R',
 'k',
 '2',
 'L',
 'p',
 'q',
 ';',
 '&',
 'X',
 'Q',
 'C',
 '8',
 '*',
 'K',
 'x',
 '(',
 '/',
 'T',
 'i',
 '.',
 'U',
 "'",
 '0',
 'j',
 'F',
 'A',
 'r',
 '+',
 'l',
 'J',
 'v',
 'N',
 'Z',
 '?',
 'D',
 'g',
 '-',
 'y',
 'I',
 'P',
 'Y',
 'h',
 's',
 'n',
 '6',
 ',',
 'c',
 'o',
 'b',
 't',
 'w',
 '!',
 'u',
 'H',
 '7',
 'G',
 ')',
 'W',
 'E',
 '3',
 'V',
 '9',
 ':'], mask_token=None, invert=True
)

# This is the mapping for the 100 epoch model
num_to_char2 = StringLookup(
    vocabulary=['[UNK]', 's', '3', 'Y', 'G', 'j', 'O', '6', ')', 'p', '-', 'I', 'F', 'P', 'd', 'Q', '!', 'M', 'o', 'T', '+', '(', '4', '&', 'f', 'h', '1', 'e', 'K', '#', 'z', 'X', 'c', 'R', 'E', 'L', '9', '2', 'i', '7', '*', 'B', '/', 'Z', 'W', '8', ';', 'r', 'N', "'", 'U', 'q', ',', 'a', '5', 'J', 'n', 'H', ':', 'C', '"', 'k', '.', 'v', 'S', 'V', 'y', 'l', 'x', 'D', '?', 'g', 'w', 'u', '0', 'b', 'A', 'm', 't'],
    mask_token=None, invert=True
)

# Comments below just show some code used for iterative testing and visualisation of the functionality of the above functions
"""
# Code here runs model predictions and then shows a grid with the image of a word and the model's prediction
# Gives a good visualization of the model's performance - can see image of word and what model predicted it to be
test_path = "data/sample1.png"
extracted_words = extractWords(test_path)

tensor_images = list()

for extracted_word in extracted_words:
    tensor_images.append(preprocessImage(extracted_word))

tensor_images = tf.convert_to_tensor(tensor_images, dtype=tf.float32)


model_path = 'htr_models/htr_model_30epochs.keras'
new_model = tf.keras.models.load_model(model_path)
preds = new_model.predict(tensor_images)
pred_texts = decode_predictions(preds)
print(pred_texts)

size = 4
_, ax = plt.subplots(size, size, figsize=(30, 14))
for i in range(size**2):
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
"""
