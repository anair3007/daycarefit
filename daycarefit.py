#!/usr/bin/python
import flask, sklearn, os, string, re, gensim
from flask import Flask, render_template, flash, request, redirect, url_for
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

from string import digits
punctuations = string.punctuation

app = Flask(__name__)
app.config.from_object(__name__)

def process_string_to_normvector(sentence):
	model = gensim.models.KeyedVectors.load('data/model-w2v-normalized-read-only')
	model.init_sims(replace=True)
	#remove trailing spaces + lowercase
	processed = sentence.strip().lower().split()
	#remove numbers, stopwords, and punctuations
	processed = [token.translate({ord(k): None for k in digits}) for token in processed if(token not in ENGLISH_STOP_WORDS and token not in punctuations)]
	remove = ['th','rd','st','']
	processed = [re.sub(r'\W', '', token).strip() if hasattr(token,'strip') else re.sub(r'\W', '', token) for token in processed]
	processed = [token for token in processed if (token not in remove and token in model.vocab)]
	if(len(processed)==0):
	    return None
	else:
	    processed = np.mean([model.word_vec(token) for token in processed],axis=0)
	    return processed/np.linalg.norm(processed)

def process_input(sentence,city):
	location = get_location(city)

	processedinput = process_string_to_normvector(sentence)
	if processedinput is None:
		return None, location
	else:
		df = joblib.load('data/df_norm_vector')
		df = df.loc[(df['city']==location)& (df['biz-ratingcount'] != '1 review') & (df['biz-ratingcount'] != '2 reviews') ]
		sims=[cosine_similarity(processedinput.reshape(1,-1),df.at[i,'vector']) for i in df.index]
		df['similarity'] = sims
		df['similarity']=df['similarity'].astype(float).apply(lambda x: 100.*x)
		df=df.sort_values(by='similarity',ascending=False).reset_index(drop=True)
		return df[:4], location

def get_location(city):
	city_names = {"NYC":"New York, NY", "LA":"Los Angeles, CA"}
	return city_names[city]

@app.template_filter('nl2br')
def nl2br(s):
	return s.replace("\n", "<br />")

@app.template_filter('nan')
def not_available(s):
	return str(s).replace("nan", "Not Available")

@app.route('/', methods=['GET'])
def index():
	return render_template('index.html')

@app.route('/location', methods=['GET','POST'])
def location():
	if request.method == 'GET':
		city = request.args.get('city', '')
		location= get_location(city)
	return render_template('location.html', city=city,location=location)

@app.route('/match', methods=['POST','GET'])
def match():
	error=None
	if request.method=='POST':
		sentence = request.form.get('statement1')+', '+request.form.get('statement2')+', '+request.form.get('statement3')
		city = request.args.get('city', '')
		matches,location = process_input(sentence, city)
		if (matches is None):
			error = 'Whoops! We did not recognize any of those words, try again.'
			return render_template('location.html', city=city,location=location,error=error)
	return render_template('match.html',matches=matches,sentence=sentence, city=city,location=location)

if __name__ == '__main__':
	app.run(host='0.0.0.0',debug=True)
