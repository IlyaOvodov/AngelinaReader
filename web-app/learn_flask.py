from flask import Flask, render_template, request

import sys
sys.path.insert(1, '..')
sys.path.insert(2, '../NN/RetinaNet')
import os
import infer_retinanet


IMG_ROOT = 'static/upload'

recognizer = infer_retinanet.BrailleInference()

app = Flask(__name__)

from flask_uploads import UploadSet, configure_uploads, IMAGES
photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = IMG_ROOT
configure_uploads(app, photos)

#@app.route("/")
#def index():
#    return render_template('home.html')
 
@app.route("/aboutus")
def aboutus():
    return render_template('aboutus.html')

@app.route("/tutorial_slider")
def tutorial_slider():
    return render_template('tutorial_slider.html')

@app.route("/")
def upload():
    return render_template('upload.html')

@app.route("/upload_results", methods=['GET', 'POST'])
def save():
    
    # Save the image in the path
    if request.method == 'POST' and 'fileField' in request.files:
        filename = photos.save(request.files['fileField'])
    
    img_path = IMG_ROOT + "/" + filename
    out_img, lines, out_text = recognizer.run(img_path)
    out_filename = img_path+'.labeled.jpg'
    out_img.save(out_filename)

    with open(img_path+'.labeled.txt', 'w') as f:
        for s in out_text:
            f.write(s)
            f.write('\n')

    #full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'shovon.jpg')

    return render_template('display.html', filename=out_filename, letter=out_text)

if __name__ == "__main__":
    debug = False

    if debug:
        app.run(debug=True)
    else:
        app.run(host='0.0.0.0')
