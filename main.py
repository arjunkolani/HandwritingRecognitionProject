from flask import Flask, render_template, flash, request, redirect
from werkzeug.utils import secure_filename
import os
from inference import convert_text, spell_correct
from wordSegmentation import drawWords
import cv2

app = Flask(__name__)
# Set app configurations
app.config["SECRET_KEY"] = "secretKey"
app.config["UPLOAD_FOLDER"] = "static/files"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["png", "jpg", "jpeg"]


def allowed_image_extension(filename: str) -> bool:
    """
        Function takes a filename and returns if it is valid for an image file we accept.

        Paramaters:
        - filename (str): The filename to check.
        
        Returns:
        - bool: True if the filename is an image file of an allowed extension, False otherwise.
    """
    if not "." in filename:
        # Checks if there's a "." in  filename - required for a valid image ifle
        return False

    extension = filename.rsplit(".", 1)[1] # Splits into array of text before and after "."
    # Then we access the second element of the array to get our file extension

    if extension.lower() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


# Gets path of folder to store images in
folder_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'])
@app.route('/', methods = ['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    """
        Flask route for home page, accepts GET and POST requests
    """
    if os.path.isfile(os.path.join(folder_path, "segmented_image.png")):
        # Checks if our word segmented image file exists from a previous conversion
        # We need to delete it to prevent conflicts when we try to open a different image saved as the same name
        # Also prevents folder/back-end server from becoming clogged up
        os.remove(os.path.join(folder_path, "segmented_image.png"))

    if request.method == "POST":
        if request.files:
            # Code for post request with a file uploaded
            image = request.files["image"]

            if image.filename == "":
                # Validation to ensure image is uploaded
                #print("No image was uploaded")
                flash("No image was uploaded")
                return redirect(request.url)

            if not allowed_image_extension(image.filename):
                # Validation to ensure image is of a valid file type
                #print("Invalid image extension")
                flash("Invalid image extension")
                return redirect(request.url)

            # Creates file path to store file
            file_path = os.path.join(folder_path, secure_filename(image.filename))
            # If file already exists somehow from previous uploading, we delete it
            if os.path.isfile(file_path):
                os.remove(file_path)

            # Saves our image to the required file path
            image.save(file_path)

            # Predict text using model
            output_text = convert_text(file_path, "htr_models/new_htr_model_100epochs.keras")

            # Get spell-corrected text
            spelling_text = spell_correct(output_text)

            # Get image with boxes drawn around each word and store it in our static folder so it can be displayed
            segmented_image = drawWords(file_path)
            os.remove(file_path)  # To stop files folder clogging up
            # Saves segmented image file so that we can display it on web-page after conversion
            cv2.imwrite(os.path.join(folder_path, "segmented_image.png"), segmented_image)

            return render_template("home.html", output_text=output_text, status="Converted",
                                   spelling_text=spelling_text)


    return render_template("home.html", output_text="", status="Default", spelling_text="")



if __name__ == '__main__':
    app.run() # Runs our flask application