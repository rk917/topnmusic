from flask import Flask, render_template, request
import os
import runModel


UPLOAD_FOLDER = os.getcwd()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/how_it_works')
def how_it_works():
    return render_template('how_it_works.html')

@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == 'GET':
        return render_template('results.html')

    if request.method == 'POST':

        # Read in the input file
        inputFile = request.files['file']
        inputFile.save(os.path.join(app.config['UPLOAD_FOLDER'], inputFile.filename))
        filename = inputFile.filename

        runModel.mp3ToWav(filename)

        prediction = runModel.main('./temp.wav')
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], './temp.wav'))

        return render_template('results.html', prediction_text=prediction, filename=filename)

if __name__ == "__main__":
    app.run()

