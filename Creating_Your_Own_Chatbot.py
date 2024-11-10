import nltk


from nltk.tag import pos_tag

from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.corpus import stopwords, wordnet

from nltk.stem import WordNetLemmatizer

import string

import pandas as pd

import streamlit as st

df0=pd.read_csv('DataScience QA.csv')
df1=pd.read_csv('DataScienceBasics_QandA - Sheet1.csv')
df1.drop('Id',axis=1,inplace=True)

df=pd.concat([df0,df1],axis=0)
df.reset_index(inplace=True)
df.drop_duplicates(inplace=True)
dfQ=df['Question']
dfA=df['Answer']
sentences=dfQ.apply(lambda x:sent_tokenize(x))
sentences=sentences.to_list()

def preprocess(sentence) :

    # Tokenize the sentence into words (Tokenisation de la phrase en mots)

    words = word_tokenize(sentence)

    # Suppression des mots vides et de la ponctuation

    words = [word.lower() for word in words if word.lower() not in stopwords.words('english') and word not in string.punctuation]

    # Lemmatisation des mots

    lemmatizer = WordNetLemmatizer()

    words = [lemmatizer.lemmatize(word) for word in words]

    return words


sentences = [str(sentence) for sentence in sentences]

# Prétraitement de chaque phrase du texte

corpus = [preprocess(sentence) for sentence in sentences]

# Définir une fonction pour trouver la phrase la plus pertinente en fonction d'une requête

def get_most_relevant_sentence(query) :

    # Prétraitement de la requête

    query = preprocess(query)

    # Calcule la similarité entre la requête et chaque phrase du texte

    index=0
    max_similarity = 0

    most_relevant_sentence = ""

    for sentence in corpus :

        similarity = len(set(query).intersection(sentence)) / float(len(set(query).union(sentence)))

        if similarity > max_similarity :

            index=corpus.index(sentence)
            max_similarity = similarity

            most_relevant_sentence = sentence


    return most_relevant_sentence,index


def chatbot(question) :

    # Trouver la phrase la plus pertinente

    phrase_la_plus_pertinente,index = get_most_relevant_sentence(question)


    if phrase_la_plus_pertinente==phrase_la_plus_pertinente:
        
        question_answer=df.loc[[index]].values.tolist()

        Question=question_answer[0][1]

        Answer=question_answer[0][2]
    else:
        Question='?'
        Answer='Sorry, it seems that I have no answer. Please try new query'

    # Retourne la réponse

    return Question,Answer

# Créer une application Streamlit

def main() :

    st.title("Chatbot")

    st.write("Bonjour ! Je suis un chatbot. Demandez-moi n'importe quoi sur le sujet dans le fichier texte.")

    # Obtenir la question de l'utilisateur

    question = st.text_input("You:" )

    # Créer un bouton pour soumettre la question

    if st.button("Submit"):

        # Appeler la fonction chatbot avec la question et afficher la réponse

        Question,Answer = chatbot(question)

        st.write("Chatbot : Your question is: \n  "  + Question)
        st.write("Chatbot : The answer is:  \n     "  + Answer)

if __name__ == "__main__" :

    main()