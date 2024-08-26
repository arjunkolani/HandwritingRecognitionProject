from flask import Flask, render_template, flash
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired, ValidationError
from inference import convert_text

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretKey'
app.config['UPLOAD_FOLDER'] = 'static/files'

image_extensions = ['.jpg', '.jpeg', '.png']

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Convert To Text")


@app.route('/', methods = ['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
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


if __name__ == '__main__':
    print(os.path.abspath(os.path.dirname(__file__)))
    app.run(debug=True)