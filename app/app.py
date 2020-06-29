from flask import *
from feature import image_pros
import os
from make_dataset import makeinterim, makeprocessed
from extract_features import extract_features
from train_pred import train_pred

app = Flask(__name__)

# referred from https://www.javatpoint.com/flask-file-uploading
@app.route('/')
def upload():
    return render_template("options.html")

# @app.route('/location')
# def location():
#
#     return render_template('location.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        f.save('image.jpg')
    output = image_pros()
    return(output)

@app.route('/train', methods = ['POST'])
def train():
    if request.method == 'POST':
        f = request.files.getlist("my_file[]")
        for i in f :
            i.save('raw/'+i.filename)
    makeinterim()
    makeprocessed()
    extract_features ()
    table = train_pred()
    return (table)

if __name__ == '__main__':
    app.run(debug = True)
