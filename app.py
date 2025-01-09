from flask import Flask,request # type: ignore
from tools import text_to_vector  # pylint: disable=wrong-import-position
from flask import render_template # type: ignore
import pickle




app = Flask(__name__)  # pylint: disable=invalid-name

model = pickle.load(open('modelvclf.pkl', 'rb'))



@app.errorhandler(404)
def page_not_found(error):
    return 'page not found', 404


@app.errorhandler(500)
def internal_error(error):
    return 'internal error', 500


@app.errorhandler(403)
def forbidden(error):
    return 'forbidden', 403



@app.route('/request' , methods=['GET' , 'POST'])
def home():
    message=''
    my_prediction = ''
    if request.method == 'POST':
        # return 'ok'
        message = request.form['message']
        vector = text_to_vector(message)
        # my_prediction = model.predict(vect)
        prob_vclf = model.predict_proba(vector)
        if prob_vclf[0][1] > 0.1:
            print("SPAM")
            my_prediction = 'Spam'
            return render_template('request.html',  message=message, prediction=my_prediction)
        else :
            print("NOT SPAM")
            my_prediction = 'Not Spam'
            return render_template('request.html',message=message, prediction=my_prediction)
    
    else :
        return render_template('request.html' , prediction=my_prediction,message=message)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0' , port=5000)