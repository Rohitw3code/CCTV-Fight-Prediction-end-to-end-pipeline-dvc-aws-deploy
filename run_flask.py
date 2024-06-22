from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from fightClassifier.prediction import ModelPredictor
import os
import numpy as np
import imageio
import shutil
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'avi', 'mpg', 'mp4'}

model = ModelPredictor()

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def video_info(path):
    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(path)
        duration = clip.duration
        fps = clip.fps
        width, height = clip.size
        return duration, fps, (width, height)
    except Exception as e:
        print(f'can read the video data : {e}')


def convert_seconds_to_duration(seconds):
    days = seconds // 86400
    seconds %= 86400
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    duration_parts = []
    if days > 0:
        duration_parts.append(f"{int(days)} day{'s' if days > 1 else ''}")
    if hours > 0:
        duration_parts.append(f"{int(hours)} hour{'s' if hours > 1 else ''}")
    if minutes > 0:
        duration_parts.append(
            f"{int(minutes)} minute{'s' if minutes > 1 else ''}")
    if seconds > 0:
        duration_parts.append(
            f"{round(seconds)} second{'s' if seconds > 1 else ''}")
    return ", ".join(duration_parts)

def empty_folder(path):
    # Iterate over all the files and directories in the static folder
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            # Check if it's a file or directory and remove it
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')    

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
                video_file = os.path.join(
                    app.config['UPLOAD_FOLDER'], filename)
                duration, fps, (width, height) = video_info(video_file)
                file.save(video_file)

                return render_template('index.html', filename=filename,
                                       duration=convert_seconds_to_duration(
                                           duration),
                                       fps=int(fps),
                                       width=width,
                                       height=height)

        elif 'predict' in request.form:
            filename = request.form.get('filename')
            video_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(video_file):
                
                pred = model.predict(video_file=video_file)

                violence = model.load_long_video(video_file=video_file)

                all_gif_filenames = []
                empty_folder('static')
                if not os.path.exists('static'):
                    os.makedirs('static')

                for i, sample in tuple(violence):
                    gif_filename = f'static/testsample_{i}.gif'
                    with open(gif_filename, 'wb') as f:
                        imageio.mimsave(f,sample.astype("uint8"), "GIF", fps=5)
                    all_gif_filenames.append(gif_filename)

                gif_files = [f for f in os.listdir('static') if f.endswith('.gif')]
                print(all_gif_filenames)

                duration, fps, (width, height) = model.video_info()
                return render_template('index.html',
                                       filename=filename,
                                       prediction=pred,
                                       duration=convert_seconds_to_duration(
                                           duration),
                                       fps=int(fps),
                                       width=width,
                                       height=height,
                                       gifs=gif_files)

    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
