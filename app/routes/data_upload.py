import shutil, os
from app import app
from flask import render_template, redirect, request
from werkzeug.utils import secure_filename

data_folder = '/Users/senora/Desktop/Keshu-agriculture/app/data'

@app.route('/upload_data', methods=["GET", "POST"])
def upload_data():
    try:
        if request.method == 'POST':
            f = request.files['file']
            if os.path.exists(os.path.join(data_folder, secure_filename(f.filename))):
                return "Filename already in use. Change the file name and try again"
            f.save(os.path.join(data_folder, secure_filename(f.filename)))
            return 'file uploaded successfully'
        else:
            return render_template('upload_data.html')
    except Exception as e:
        print(e)
        return redirect('/')