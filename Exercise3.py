import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist, pos_tag
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Read Moby Dick file
with open('moby_dick.txt', 'r') as file:
    text = file.read()

# Tokenization
tokens = word_tokenize(text.lower())

# Stopwords filtering
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token not in stop_words]

# Parts-of-Speech (POS) tagging
pos_tags = pos_tag(filtered_tokens)

# POS frequency
pos_counts = FreqDist(tag for word, tag in pos_tags)
top_pos = pos_counts.most_common(5)

print("Top 5 POS and their counts:")
for pos, count in top_pos:
    print(pos, count)

# Lemmatization (Lemmatize all tokens)
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token, pos=get_wordnet_pos(tag)) for token, tag in pos_tags]

print("\nLemmatized tokens:")
print(lemmatized_tokens)

# Plotting frequency distribution
pos_counts.plot(30, cumulative=False)
plt.show()