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
nltk.download('omw-1.4')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))



# Extraction des caractéristiques (Bag-of-Words)
vectorizer = CountVectorizer()

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
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

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

