from flask import Flask, render_template, request

import json
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
    print("save")
    t = time.clock()
    # Save the image in the path
    if request.method == 'POST' and 'fileField' in request.files:
        filename = photos.save(request.files['fileField'])
    print(time.clock() - t)
    print("recognizer.run")
    t = time.clock()
    img_path = IMG_ROOT + "/" + filename
    raw_image, out_img, lines, out_text, data_dict = recognizer.run(img_path)
    print(time.clock() - t)
    print("save")
    t = time.clock()

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    filename_stem = filename.rsplit('.', 1)[0]

    labeled_image_filename = filename_stem + '.labeled' + '.jpg'
    json_path = RESULTS_ROOT + "/" + filename_stem + '.labeled' + '.json'
    raw_image.save(RESULTS_ROOT + "/" + labeled_image_filename)
    data_dict['imagePath'] = labeled_image_filename
    with open(json_path, 'w') as opened_json:
        json.dump(data_dict, opened_json, sort_keys=False, indent=4)

    marked_image_path = RESULTS_ROOT + "/" + filename_stem + '.marked' + '.jpg'
    recognized_text_path = RESULTS_ROOT + "/" + filename_stem + '.marked' + '.txt'
    out_img.save(marked_image_path)
    with open(recognized_text_path, 'w') as f:
        for s in out_text:
            f.write(s)
            f.write('\n')
    print(time.clock() - t)
    print(("total", time.clock() - t0) )
    #full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'shovon.jpg')


    return render_template('display.html', filename=marked_image_path, letter=out_text)

if __name__ == "__main__":
    debug = True
    app.jinja_env.cache = {}
    if debug:
        app.run(debug=True)
    else:
        app.run(host='0.0.0.0', threaded=True)
