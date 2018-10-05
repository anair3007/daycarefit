#!/usr/bin/python
import flask, spacy, sklearn, os
from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import pandas as pd
from sklearn.externals import joblib

app = Flask(__name__)
app.config.from_object(__name__)

def process_input(sentence,city):
	nlp = spacy.load('data/nlp')
	inputdoc = nlp(sentence)
	if (city=="LA"):
		city_df = joblib.load('data/la.joblib').reset_index(drop=True)
	elif (city=="NYC"):
		city_df = joblib.load('data/ny.joblib').reset_index(drop=True)

	for i in city_df.index:
		city_df.loc[i, 'similarity'] = city_df.loc[i, 'pruned'].similarity(inputdoc)
	city_df.loc[:, 'similarity'] = city_df.loc[:, 'similarity'].apply(lambda x: 100*x)

	city_df = city_df.sort_values(by='similarity', ascending=False).reset_index(drop=True)
	return city_df[:4]

def get_location(city):
	city_names = {"NYC":"New York, NY", "LA":"Los Angeles, CA"}
	return city_names[city]

@app.template_filter('nl2br')
def nl2br(s):
	return s.replace("\n", "<br />")

@app.template_filter('nan')
def not_available(s):
	return s.replace(" Nan", "Not Available")

@app.route('/', methods=['GET'])
def index():
	return render_template('index.html')

@app.route('/location', methods=['GET'])
def location():
	if request.method == 'GET':
		city = request.args.get('city', '')
		location= get_location(city)
	return render_template('location.html', city=city,location=location)

@app.route('/match', methods=['POST'])
def match():
	if request.method == 'POST':
		sentence = request.form.get('statement1')+', '+request.form.get('statement2')+', '+request.form.get('statement3')
		city = request.args.get('city', '')
		location= get_location(city)
		matches = process_input(sentence, city)
	return render_template('match.html',matches=matches,sentence=sentence, city=city,location=location)

if __name__ == '__main__':
	app.run(host='0.0.0.0',debug=False)
