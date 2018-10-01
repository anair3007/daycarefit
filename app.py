from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import pandas as pd
import spacy
import numpy as np
import gensim
import pickle
import scipy
from scipy.spatial.distance import euclidean
import sklearn
from sklearn.metrics.pairwise import euclidean_distances
import os
APP_ROOT = os.path.dirname(os.path.abspath(__file__))   # refers to application_top
APP_STATIC = os.path.join(APP_ROOT, 'static')
with open('data/df_unique_biz.pickle', 'rb') as f:
	df_unique_biz = pickle.load(f)

with open('data/lemmatized.pickle', 'rb') as f:
	documents = pickle.load(f)
#build vocabulary and train model
model = gensim.models.Word2Vec(documents,size=100,window=5,min_count=1,workers=4)
model.train(documents, total_examples=len(documents), epochs=5)

list_of_avg=[]
for i, business in enumerate(documents):
    list_of_vectors=[]
    business_mean=[]
    for word in business:
        list_of_vectors.append( np.array(model.wv.word_vec(word)))
    if(len(list_of_vectors)<1):
        print(i)
        continue
    else:
        list_of_avg.append( np.mean(list_of_vectors, axis=0) )
biz_name_list = df_unique_biz['biz-name'].values.tolist()
biz_rating_list = df_unique_biz['biz-rating'].values.tolist()
biz_ratingcount_list = df_unique_biz['biz-ratingcount'].values.tolist()
biz_phone_list = df_unique_biz['biz-phone'].values.tolist()
biz_address_list = df_unique_biz['biz-address'].values.tolist()
biz_city_list = df_unique_biz['city'].values.tolist()
biz_review_list = df_unique_biz['review-text'].values.tolist()
biz_url_list = df_unique_biz['biz-url'].values.tolist()

listToRemove = [1170,1251,1444,1460,1892,3321,3543,3613,3625,4095,4656,5462,5489]

for i in listToRemove:
    del biz_name_list[i]
    del biz_rating_list[i]
    del biz_ratingcount_list[i]
    del biz_phone_list[i]
    del biz_address_list[i]
    del biz_city_list[i]
    del biz_review_list[i]
    del biz_url_list[i]

finaldf = pd.DataFrame({'name':biz_name_list,'rating':biz_rating_list,'ratingcount':biz_ratingcount_list,'phone':biz_phone_list,
                        'address':biz_address_list,'city':biz_city_list,'url':biz_url_list,'review':biz_review_list,'vector':list_of_avg})


app = Flask(__name__)
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d4asdasd441f2b6176a'

class ReusableForm(Form):
    state = TextField('State:', validators=[validators.required()])
    city = TextField('City:', validators=[validators.required()])

def read_df():
    df = pd.read_csv('data/finaldf.csv')
    return df

def process_input(s1,s2,s3,location):
    n2=""
    n1=""
    pos_to_keep = ['JJ','JJR','JJS']
    sentence = str(s1+" "+s2+" "+s3)
    nlp = spacy.load('en', disable=['textcat','ner','parser'])
    input_token = nlp(sentence)
    input_adj = [token.text for token in input_token if token.tag_ in pos_to_keep]
    input_vector = np.mean([model.wv.word_vec(word) for word in input_adj], axis=0)

    dist_list=[]
    for i in finaldf.index:
        if (finaldf.at[i,'city'] == location):
            vA = input_vector
            vB = finaldf.at[i,'vector']
            dist_list.append(float(euclidean(vA,vB)))
    topfive = pd.Series(dist_list).nlargest(10).index
    names = finaldf.iloc[topfive]['name'].unique()
    n1 = names[0]
    n2 = names[1]

    return n1, n2


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/location', methods=['GET'])
def location():
    if request.method == 'GET':
        state = request.args.get('stateSelect','')
        city = request.args.get('citySelect', '')
        flash("You entered: " + str(city) + ", " + str(state))
    return render_template('location.html',city=city, state=state)

@app.route('/match', methods=['POST'])
def match():
    location = "New York, NY"
    if request.method == 'POST':
        s1 = request.form.get('statement1')
        s2 = request.form.get('statement2')
        s3 = request.form.get('statement3')
        n1,n2 = process_input(s1,s2,s3, location)
        flash("You matched with: " + n1 + ", " + n2)
    return render_template('match.html',n1=n1, n2=n2, location = "New York, NY"
)

if __name__ == '__main__':
    app.run()
