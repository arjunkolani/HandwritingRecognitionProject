from flask import Flask, render_template, flash, request, redirect
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired, ValidationError
from inference import convert_text, spell_correct
from wordSegmentation import drawWords
import cv2

app = Flask(__name__)
app.config["SECRET_KEY"] = "secretKey"
app.config["UPLOAD_FOLDER"] = "static/files"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["png", "jpg", "jpeg"]

image_extensions = ['.jpg', '.jpeg', '.png']

class UploadFileForm(FlaskForm):
    file = FileField("Upload File", validators=[InputRequired()])
    submit = SubmitField("Convert To Text")


def allowed_image_extension(filename):
    if not "." in filename:
        return False

    extension = filename.rsplit(".", 1)[1]

    if extension.lower() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


folder_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'])
@app.route('/', methods = ['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    """
    form = UploadFileForm()
    output_text = ""
    if form.validate_on_submit():
        file = form.file.data
        extension = os.path.splitext(file.filename)[1]
        if extension in image_extensions:
            file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                   secure_filename(file.filename))
            file.save(file_path)

            output_text = convert_text(file_path, "htr_models/new_htr_model_100epochs.keras")
            os.remove(file_path) # To stop files folder clogging up
        else:
            # Will change later to make it visible to user that they made an error
            flash("Invalid file type")

    return render_template("home.html", form=form, output_text=output_text)
    """
    if os.path.isfile(os.path.join(folder_path, "segmented_image.png")):
        os.remove(os.path.join(folder_path, "segmented_image.png"))

    if request.method == "POST":
        if request.files:
            image = request.files["image"]

            if image.filename == "":
                print("No image was uploaded")
                flash("No image was uploaded")
                return redirect(request.url)

            if not allowed_image_extension(image.filename):
                print("Invalid image extension")
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
            cv2.imwrite(os.path.join(folder_path, "segmented_image.png"), segmented_image)

            return render_template("home.html", output_text=output_text, status="Converted",
                                   spelling_text=spelling_text)


    return render_template("home.html", output_text="", status="Default", spelling_text="")



if __name__ == '__main__':
    app.run(debug=True)