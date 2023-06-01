
import tweepy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#API
api_key = "9mfFpdMOJPoD315w8R0VknrfN"
api_secret = "T5SCxHe0owTx5Z6s8ypps6a8dJygHxrfZOS9RACZ8Lvt6XtLDj"
bearer_token = r"AAAAAAAAAAAAAAAAAAAAAApNnwEAAAAAghcNtKWQvVRBkfoPpViHHOYrO74%3DNFRFoLFx01Ccg160CppI2Iei1hQMc6pIDzd0KkOPdAUcjVzbio"
access_token = "1657095561839796245-jTc20bK4q8afUeAPPkJJTMOmbntnu6"
access_token_secret = "Dl42IgEXXP4MFx2H7qb9dS8Qh881NfQkWJ1ZKYX4XqeLX"

# Authentification avec les clés d'API
auth = tweepy.OAuthHandler(api_key, api_secret)
auth.set_access_token(access_token, access_token_secret)

# Créer une instance de l'API Twitter
api = tweepy.API(auth)


#Model
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Diviser les données en commentaires et classifications
def preprocess_text(text):
    # Conversion en minuscules
    text = text.lower()
    
    # Suppression des caractères spéciaux et de la ponctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenization des mots
    tokens = word_tokenize(text)
    
    # Suppression des mots vides (stop words)
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatisation des mots
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Reconstitution du texte prétraité
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Extraction des caractéristiques (Bag-of-Words)
vectorizer = CountVectorizer()

def train_model(data):
    commentaires = data['Commentaires_des_clients'].apply(preprocess_text)
    classifications = data['classification']

    X = vectorizer.fit_transform(commentaires)

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, classifications, test_size=0.2, random_state=42)

    # Construction et entraînement du modèle Random Forest
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Évaluation du modèle
    '''y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)'''

    return model

def classify_text(text, model, vectorizer):
    # Prétraitement du texte d'entrée
    preprocessed_text = preprocess_text(text)
    input_text_vect = vectorizer.transform([preprocessed_text])

    # Prédiction de la classification du texte d'entrée
    prediction = model.predict(input_text_vect)
    return prediction[0]

# Charger le DataFrame contenant les commentaires et leurs classifications
df = pd.read_csv(r'/Users/hp/Downloads/classification des commentaires - Sheet1 (3).csv')
df = df.dropna(subset=['Commentaires_des_clients'])
model = train_model(df)

def generate_response(comment):
    class_responses = {
        "Expiration date": "Hi there! Our products have been tested to have at least a 3-years shelf life if unopened and stored at normal room temperature, during which they will retain their integrity and perform as expected. If your product looks and smells the way it normally does, it is good to use. Certain products, for example sunscreens, may have an expiration date printed on the product.",
        "Delivery": "Hi there! We offer worldwide shipping for La Roche-Posay Effaclar Adapalene Gel 0.1% Acne Treatment. Please provide your shipping address at checkout and we will ensure your order is delivered to you as soon as possible.",
        "Usage instructions": "Hi there! Our product is used to treat and prevent acne, unclog pores, and reduce the appearance of blackheads and whiteheads. It is specifically designed for individuals with oily and acne-prone skin.",
        "Method of use": "Hi there! To use La Roche-Posay Effaclar Adapalene Gel 0.1% Acne Treatment, start with a clean face and apply a thin layer of the gel all over the face. It's important to avoid the eye area and any areas of broken skin. It's typically recommended to use the product once daily, preferably at night, and to follow up with a moisturizer if needed. Always follow the directions on the product label and consult with a dermatologist if you have any concerns or questions about proper usage.",
        "Compatible products for simultaneous use": "Hi there! It is generally recommended to avoid using other acne treatments or products containing salicylic acid, benzoyl peroxide, or alpha-hydroxy acids while using La Roche-Posay Effaclar Adapalene Gel 0.1% Acne Treatment. However, it is always best to consult with a dermatologist or healthcare provider for personalized recommendations on products that can be used in conjunction with this treatment."
    }

    prediction1 = classify_text(comment, model, vectorizer)
    if prediction1 is None:
        return None
    else:
        response = class_responses.get(prediction1)
    return response


# Example usage:
'''text = "What is the estimated delivery time for this product?"
response= generate_response(text)
print(response)'''


# Récupérer les commentaires liés à un tweet spécifié
tweet_id = "1663203413323141122"
comments = api.search(q=f"to:{tweet_id}") 

# Générer des réponses pour chaque commentaire
for comment in comments:
    comment_text = comment.text
    response = generate_response(comment_text)
    api.update_status(status=response, in_reply_to_status_id=comment.id)


# Example usage:
'''# Question de l'utilisateur
question = "Does this product have a best before date?"

# Appel de la fonction generate_response pour générer une réponse à la question
reponse = generate_response(question)

print("Question :", question)
print("Réponse générée :", reponse)'''