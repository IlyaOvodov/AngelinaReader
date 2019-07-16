from flask import Flask, render_template, request

import time
import os
import sys
sys.path.insert(1, '..')
sys.path.insert(2, '../NN/RetinaNet')
import infer_retinanet


IMG_ROOT = 'static/upload'
RESULTS_ROOT = 'static/results'

print("infer_retinanet.BrailleInference()")
t = time.clock()
recognizer = infer_retinanet.BrailleInference()
print(time.clock()-t)

app = Flask(__name__)

from flask_uploads import UploadSet, configure_uploads, IMAGES
photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = IMG_ROOT
configure_uploads(app, photos)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/upload_results", methods=['GET', 'POST'])
def save():
    t0 = time.clock()
    print("save input file")
    t = time.clock()
    # Save the image in the path
    filename = None
    if request.method == 'POST' and 'fileField' in request.files:
        filename = photos.save(request.files['fileField'])
    print("save input file", time.clock() - t)
    img_path = IMG_ROOT + "/" + filename
    marked_image_path, out_text = recognizer.run_and_save(img_path, RESULTS_ROOT, draw_refined = recognizer.DRAW_BOTH)
    print(("total", time.clock() - t0) )
    return render_template('display.html', filename=marked_image_path, letter=out_text)

if __name__ == "__main__":
    debug = True
    app.jinja_env.cache = {}
    if debug:
        app.run(debug=True)
    else:
        app.run(host='0.0.0.0', threaded=True)
