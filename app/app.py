from flask import *
from feature import image_pros
import os
from make_dataset import makeinterim, makeprocessed
from extract_features import extract_features
from train_pred import train_pred
import pandas as pd
import requests

app = Flask(__name__)

# referred from https://www.javatpoint.com/flask-file-uploading
@app.route('/')
def upload():
    try:
        for i in os.listdir('interim/'):
            os.remove('interim/'+str(i))
        for i in os.listdir('output/'):
            os.remove('output/'+str(i))
        for i in os.listdir('processed/training/'):
            os.remove('processed/training/'+str(i))
        for i in os.listdir('processed/evaluation/'):
            os.remove('processed/evaluation/'+str(i))
        for i in os.listdir('processed/validation/'):
            os.remove('processed/validation/'+str(i))
    except :
        None
    return render_template("options.html")

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
    report = train_pred()
    df = pd.DataFrame(report).transpose()
    return render_template('classification_report.html', tables=[df.to_html()])

if __name__ == '__main__':
    app.run(debug = True, threaded=False)
