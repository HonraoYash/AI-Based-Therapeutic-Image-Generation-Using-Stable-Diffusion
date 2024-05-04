"""from flask import Flask 
app = Flask(__name__) 
 
@app.route('/') 
def hello_world(): 
    return 'Hello, World!' 
 
if __name__ == '__main__': 
    app.run(debug=True) 

"""

from flask import Flask, render_template, send_file
from io import BytesIO

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/generate_image')
def generate_image():
    # Your code to generate or fetch the image goes here
    # For demonstration purposes, let's assume you have the image file 'image.png' in the same directory
    image_path = 'image.jpg'
    return send_file(image_path, mimetype='image/jpg')

if __name__ == '__main__':
    app.run(debug=True)
