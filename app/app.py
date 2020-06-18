from flask import *
from feature import image_pros
app = Flask(__name__)

# referred from https://www.javatpoint.com/flask-file-uploading
@app.route('/')
def upload():
    return render_template("file_upload_form.html")

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
    output = image_pros()
    return(output)

if __name__ == '__main__':
    app.run(debug = True)
