from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import pandas as pd
import numpy as np
import gensim, spacy, scipy, sklearn, os
from sklearn.externals import joblib
# !python -m spacy init-model en "data/model-spacy" --vectors-loc "data/model-w2v-vectors"

APP_ROOT = os.path.dirname(os.path.abspath(__file__))   # refers to application_top
APP_STATIC = os.path.join(APP_ROOT, 'static')
app = Flask(__name__)
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d4asdasd441f2b6176a'

def process_input(sentence,city):
	nlp = spacy.load('data/nlp')
	inputdoc = nlp(sentence)
	print(inputdoc)
	if (city=="LA"):
		city_df = joblib.load('data/la.joblib').reset_index(drop=True)
	elif (city=="NYC"):
		city_df = joblib.load('data/ny.joblib').reset_index(drop=True)

	for i in city_df.index:
		city_df.loc[i, 'similarity'] = city_df.loc[i, 'pruned'].similarity(inputdoc)
	city_df.loc[:, 'similarity'] = city_df.loc[:, 'similarity'].apply(lambda x: 100*x)

	city_df = city_df.sort_values(by='similarity', ascending=False).reset_index(drop=True)
	return city_df[:4]

@app.template_filter('nl2br')
def nl2br(s):
	return s.replace("\n", "<br />")

@app.route('/', methods=['GET'])
def index():
	return render_template('index.html')

@app.route('/location', methods=['GET'])
def location():
	if request.method == 'GET':
		city = request.args.get('city', '')
		if(city=="NYC"):
			location="New York, New York"
		elif(city=="LA"):
			location="Los Angeles, California"
	return render_template('location.html', city=city,location=location)

@app.route('/match', methods=['POST'])
def match():
	if request.method == 'POST':
		sentence = request.form.get('statement1')+', '+request.form.get('statement2')+', '+request.form.get('statement3')
		city = request.args.get('city', '')
		location=request.args.get('location', '')
		matches = process_input(sentence, city)
	return render_template('match.html',matches=matches,sentence=sentence, city=city,location=location)

if __name__ == '__main__':
	app.run()
