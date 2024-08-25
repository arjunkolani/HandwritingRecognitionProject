from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
from inference import convert_text

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretKey'
app.config['UPLOAD_FOLDER'] = 'static/files'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Convert To Text")


@app.route('/', methods = ['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename))
        file.save(file_path)

        output_text = convert_text(file_path, "htr_models/htr_model_30epochs.keras")
        os.remove(file_path) # To stop files folder clogging up
        return output_text


    return render_template("home.html", form=form)


if __name__ == '__main__':
    app.run(debug=True)