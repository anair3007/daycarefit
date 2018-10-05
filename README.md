## daycarefit: helping you find daycares that fit your needs

### Under the hood
1. Over 52,000 daycare reviews were collected using a custom scraper built with Selenium and Beautiful Soup
2. The raw data was processed using pandas, NLTK, and scikit-learn
3. Gensim's implementation of word2vec was used on the processed data to create a domain-specific model. Word2Vec's shallow neural network was trained using 150 dimensional vectors over 12 epochs using the skip-gram method. This generated a vocabulary matrix of [17,000 vectors x 150 dimensions]
3. This model was loaded into spacy and deployed on the web. The interface asks users to input statements about their ideal daycare. These statements are fed into SpaCy to create an average word vector, which is then compared to every review in their city. Currently we have Los Angeles, CA and New York, NY included. SpaCy uses "soft" cosine similarity to select the top 4 matches.

### Algorithm Considerations
1. Latent Dirichlet Allocation and TF-IDF+Regression techniques were ruled out as being effective due to the nature of daycare reviews. This yielded poor topic cohesion and heavily weighting words that were not universal to the domain of daycare quality.
2. Currently the model will produce what a user is "looking for" and not currently effective with more complex statements such as "I don't want it to be dirty." To implement this, we can use Spacy's Phraser to construct bigrams and larger n-grams to build in this complexity. This is planned for a future release.