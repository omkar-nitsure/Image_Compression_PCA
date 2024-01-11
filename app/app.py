from flask import Flask, jsonify, request, send_file, flash, redirect
from werkzeug.utils import secure_filename
import os

from utilities import compress_img

app = Flask(__name__)

@app.route('/compress')
def compress():

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    filename = secure_filename(file.filename)
    file.save(os.path.join('/inputs/', filename))
    compress_img(filename)

    return send_file("output.png", mimetype='image/png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)