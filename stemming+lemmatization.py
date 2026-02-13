"""
👨‍💻 Stemming Task

Your goal is to:
1️⃣ Read the content of either story1.txt or story2.txt.
2️⃣ Apply regex with re.sub() to remove:
    - HTML tags (e.g. <div>...</div>)
    - URLs
    - Hashtags (#), asterisks (*), excessive punctuation (e.g. !!!, ???)
    - Extra whitespace

3️⃣ Tokenize the cleaned text into words using nltk.word_tokenize.
4️⃣ Remove stopwords using nltk.corpus.stopwords.
5️⃣ Apply stemming using nltk.stem.PorterStemmer to reduce each word to its root form.
6️⃣ Print out the list of stemmed words.

📌 Hints:
- Remember to import the required NLTK modules.
- Think about what patterns to use in your regex for URLs and HTML tags.
- Inspect intermediate results to ensure your cleaning is working!

👨‍💻 Lemmatization Task

Your goal is to:
1️⃣ Read the content of either story1.txt or story2.txt.
2️⃣ Apply regex with re.sub() to remove:
    - HTML tags (e.g. <div>...</div>)
    - URLs
    - Hashtags (#), asterisks (*), excessive punctuation (e.g. !!!, ???)
    - Extra whitespace

3️⃣ Tokenize the cleaned text into words using nltk.word_tokenize.
4️⃣ Remove stopwords using nltk.corpus.stopwords.
5️⃣ Tag each word with its part of speech using nltk.pos_tag.
6️⃣ Map POS tags to WordNet tags so the lemmatizer can use them.
7️⃣ Apply lemmatization using nltk.stem.WordNetLemmatizer, passing the correct POS.
8️⃣ Print out the list of lemmatized words.

📌 Hints:
- You’ll need a helper function to convert Treebank POS tags to WordNet POS tags.
- Check your intermediate outputs (POS tags, lemmatized results).

Write your code below this string.
"""
import nltk
import re
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords 
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# 1. Open and read the story text - CAN CHANGE HERE EASILY 1, 2 OR CUSTOM!
with open("story2.txt", "r", encoding="utf-8") as file:
    story = file.read()

# 2. Remove any unwanted characters using re.sub
clean_story = re.sub(r'http\S+|<.*?>|[’]|[‘]|[“]|[”]|[^\w\s]', '', story).lower()

# http\S every http:
# <.*?> every HTML Tag
# [^\w\s] everything that is not a letter

# 3. Tokenize the story into sentences
sentences = [sent_tokenize(story)]

# 4. Tokenize the story into words
words = [word_tokenize(story)]

# 5. Print results
print("\n=== Sentences ===")
print(sentences, "\n")
print("=== Words ===")
print(words, "\n")

story_tokenized_by_word = word_tokenize(clean_story)
print("=== Clean Word Count ===")
print(len(story_tokenized_by_word), "\n")

stop_words = set(stopwords.words("english"))
stopwords_removed = [word for word in story_tokenized_by_word if word not in stop_words]
print("=== Stopwords Removed ===")
print(len(stopwords_removed), "\n")

stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(w) for w in stopwords_removed]
print("=== Stemmed Words ===")
print(stemmed_words, "\n")

lemmatizer = WordNetLemmatizer()

# Lemmatize each word 
lemmatized_words = [lemmatizer.lemmatize(word) for word in stopwords_removed]
print("=== Lemmatized Words ===")
print(lemmatized_words, "\n")

# Tag each word with its part of speech
tagged_words = pos_tag(stopwords_removed)
print("=== Tagged Words ===")
print(tagged_words, "\n")

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun if unknown
    
lemmatized_words_with_pos = [
lemmatizer.lemmatize(word, get_wordnet_pos(pos_tag))
    for word, pos_tag in tagged_words
]
print("=== Lemmatized Words With Pos ===")
print(lemmatized_words_with_pos, "\n")