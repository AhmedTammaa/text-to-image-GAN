import tensorflow_hub as hub
import tensorflow_text
import pandas as pd
import pickle
import re
import numpy as np
from tqdm import tqdm
import string
import nltk
from nltk.corpus import stopwords
import nltk
class EmbeddingCreator:
    
    def __init__(self):
        preprocess_url = 'bert_en_uncased_preprocess_3'
        encoder_url = 'bert_en_uncased_L-12_H-768_A-12_4'
        self.bert_preprocess_model = hub.KerasLayer(preprocess_url)
        self.bert_model = hub.KerasLayer(encoder_url)
    
    def create_image_csv(self):
        rootdir = 'images/birds/CUB_200_2011/images'
        df = pd.DataFrame(columns=['images'])
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith(".jpg") or filepath.endswith(".jpeg") or filepath.endswith(".png"):
                    df.loc[len(df)] = [file]
        df.to_csv("intermediate_results.csv", index=False)

    def create_final_csv(self):
        rootdir = '/captions/text_c10'
        images = pd.read_csv('intermediate_results.csv')
        images['captions'] = 'abc'
        print(images.head())
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith(".txt"):
                    with open(filepath) as f:
                        content = f.readlines()
                    content = [x.strip() for x in content]
                    t = file.replace('.txt', '.jpg')
                    images.loc[images['images'] == t, 'captions'] = content[random.randint(0, 1)]
        images.to_csv("final.csv", index=False)

    
    def create_feature_vectors_for_single_comment(self,model, cleaned_comments, image_names):
        vectorized_list = []
        image_list = []
        print(type(image_names))
        print(len(cleaned_comments))
        for comments, image in tqdm(zip(cleaned_comments, image_names)):
            result_array = np.empty((0, 768))
            for word in comments:
                #print(word)
                w = self.bert_preprocess_model([word])
                w = model(w)
                #w = model(w)
                w = w['pooled_output']
                #print(len(w))
                #print(w.numpy())
                result_array = np.append(result_array, w, axis=0)

            vectorized_list.append(np.mean(result_array, axis=0).astype('float32'))
            image_list.append(image)

        return image_list, np.array(vectorized_list)

    def cleanDataFrame(self, df):
        stop = stopwords.words('english')
        df['captions'] = df['captions'].str.replace('[^\w\s]','',regex=True)
        df['captions'] = df['captions'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        return df
    
    def createEmbeddingFirstTime(self, file_name= '1_test_word_vector_min_bird'):
        df = pd.read_csv('final.csv')
       # df = df[1:20]
        model = self.bert_model
        df = self.cleanDataFrame(df)
        cleaned_captions = df['captions'].values
        
        image_names = df['images'].values
        #print(type(image_names))
        #print(image_names)
        #print(image_names.shape)
        #print('Done tokenizing....')
        i, c = self.create_feature_vectors_for_single_comment(model, cleaned_captions, image_names)
        word_vector_dict = dict(zip(i, c))
        pickle.dump(word_vector_dict, open(file_name + ".p", "wb"))
        #print('Done')
        #print(i)
        #print()
        return word_vector_dict
    
    
    def cleanString(self, sen):
        stop = stopwords.words('english')
        sen = sen.lower()
        sen = sen.translate(str.maketrans('','',string.punctuation))
        sen = ' '.join([word for word in sen.split() if word not in (stop)])
        return sen

    def createCustomEmbeddings(self, sentence):
        sentence  = self.cleanString(sentence)
        sentence_list = [sentence]
        cleaned_captions = np.array(sentence_list)
        image_names = np.array(sentence_list)
        #print(type(image_names))
        
        #print(image_names)
        #print(image_names.shape)
        #print('Done tokenizing....')
        i, c = self.create_feature_vectors_for_single_comment(self.bert_model , cleaned_captions, image_names)
        word_vector_dict = dict(zip(i, c))
        
        return c