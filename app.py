from flask import Flask,render_template,request
import pickle
import nltk
import string
from nltk.corpus import stopwords
sw = stopwords.words("english")
punc = string.punctuation
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


#initialize the app
app = Flask(__name__)

#initalize the vectorizer
tfidf = pickle.load(open('C:\\Users\\siddh\\SMS Spam Classifier\\vectorizer.pkl','rb'))


#initialize the model
model = pickle.load(open('C:\\Users\\siddh\\SMS Spam Classifier\\model.pkl','rb'))

#transform function
def transform_text(text):
    
    #lowercase
    text = text.lower()
    
    #Word Tokenization
    text = nltk.word_tokenize(text)
    
    # Removing special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y.copy()
    y.clear()
        
    # Removing stopwords
    for i in text:
        if i not in sw and i not in punc:
            y.append(i)
            
    text = y.copy()
    y.clear()
    
    #Stemming
    for i in text:
        y.append(ps.stem(i))
        
    #output
    return " ".join(y)    




@app.route("/")
def model_predict():
    return render_template("sms.html")

@app.route("/input" ,methods=['post'])
def take_user_input():
    sms = request.form.get("text")

    input_sms = sms

    #preprocess
    transformed_sms = transform_text(input_sms)

    #vectorize
    vector_input = tfidf.transform([transformed_sms])

    #predict
    result = model.predict(vector_input)[0]

    #display
    if result == 1:
        return "Spam"

    else:
        return "Not Spam"
   

app.run(debug=True)