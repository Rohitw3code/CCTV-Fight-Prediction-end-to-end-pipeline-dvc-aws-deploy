from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from fightClassifier.prediction import ModelPredictor
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'avi', 'mpg', 'mp4'}

model = ModelPredictor()

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'upload' in request.form:
            if 'file' not in request.files:
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = file.filename
                video_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(video_file)
                return render_template('index.html', filename=filename)
        elif 'predict' in request.form:
            filename = request.form.get('filename')
            video_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(video_file):
                pred = model.predict(video_file=video_file)
                return render_template('index.html', filename=filename, prediction=pred)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
