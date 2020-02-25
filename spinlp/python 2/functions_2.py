# Author: Sumedh Chandaluri
# 206021

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV, train_test_split, validation_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression, SGDClassifier

from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle


class SPi_basic_nlp:

    def __init__(self, df, text_col, class_col, max_features, n_gram, name, path):
        
        """ self, represents the instance of the class.
            
            Here by creating an object, you can have train-test-split, then create a count vectorizer and 
            fit_transform on train set & transform on test set. After that we save count vectorizer object for tranforming on test query also
            
            Inputs
            1. DataFrame
            2. Text Column Name
            3. Class Label Column Name
            4. max_features to be used for count_vectorizer (sklearn)
            5. n-grams to be used for count_vectorizer (sklearn)
            6. each type of 'strtaxonomy'(a column in df)
                eg: We have 3 different types of 'strtaxonomy'
                    a) for 'patient problem code', keep name = 'patient'
                    b) for 'device problem code', keep name = 'device'
                    c) for 'medical evaluation result code', keep name = 'medical'
            7. path where we save all our 
                a) count_vectorizer object for each 'strtaxonomy'
                b) model for each 'strtaxonomy'
        """
        
        self.df = df
        self.text_col = text_col
        self.class_col = class_col
        self.max_features = max_features
        self.name = name
        self.path = path
        self.ngram = n_gram
        X_train_df, X_test_df, self.y_train, self.y_test = train_test_split(df[text_col], df[class_col], test_size=0.1, random_state=93)
        with open('stopwords_2.pickle', 'rb') as handle:
            stopwords = pickle.load(handle)
        
        # count_vec = CountVectorizer(ngram_range=(1, n_gram), binary=True, max_features=max_features, stop_words=stopwords)
        # self.X_train = count_vec.fit_transform(X_train_df)
        # self.X_test = count_vec.transform(X_test_df)       
        # with open(self.path + 'count_vectorizer_' + self.name + '.pickle', 'wb') as handle:
            # pickle.dump(count_vec, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        tfidf_vec = TfidfVectorizer(ngram_range=(1, n_gram), binary=True, max_features=max_features, stop_words=stopwords)
        self.X_train = tfidf_vec.fit_transform(X_train_df)
        self.X_test = tfidf_vec.transform(X_test_df)
        with open(self.path + 'tfidf_vectorizer_' + self.name + '.pickle', 'wb') as handle:
            pickle.dump(tfidf_vec, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def basic_preprocessing(cls, df, text_cols):
        
        """ Basic Pre-Processings like
            1. Making all given text columns into lower
            2. Removing spcaes/new line at the end of the text
            eg: a) 'cats ' to 'cats'
                b) 'dogs\n' to 'dogs'
            3. Reseting Index
                
            Inputs
            1. DataFrame
            2. All the columns (list of strings/column names) where text is present and want to remove spaces/new line at the end.
            
            Output
            1. Pre-processed DataFrame
        """

        for col in text_cols:
            df[col] = df[col].str.lower()
            df[col] = df[col].str.rstrip()
        
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return df
    
    @classmethod
    def stemming_sentence(cls, sentence):
        
        """ Given a string, it converts each word into it's root word and returns a stemmed string.
            You have to use below code to apply on the whole column of a DataFrame:
            
            'df[column name] = df[column name].apply(SPi_basic_nlp.stemming_sentence)'
            
            Input:
                1. string
            Output:
                1. string
        """
        porter_stemmer = PorterStemmer()
        tokens = sentence.split()
        stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
        
        return ' '.join(stemmed_tokens)
    
    @classmethod    
    def plural_to_singular(cls, content):
        
        ''' Given a string, it converts each plural word into it's singular form and returns a string.
            You have to use below code to apply on the whole column of a DataFrame:
            
            'df[column name] = df[column name].apply(SPi_basic_nlp.plural_to_singular)'
            
        Inputs:
            content: string 
        Returns:
            content: plurals in the string (content) made to singular
        '''
        blob = TextBlob(content)
        singles = [word.singularize() for word in blob.words]                              # plural to singular
        
        return ' '.join(singles)

    @classmethod
    def after_preprocessings(cls, df, str_1, top):
        
        """ Making the data feedable to create a Model
            1. Calculating top 'n' most occuring class labels and making all other variables as 'others'((n+1)th class)
            2. Returns a DataFrame
            
            Inputs
            1. DataFrame
            2. type of DataFrame
            3. 'n' most occuring class labels
            
            Output
            1. DataFrame making remaining labels as 'others'
        """
        
        new = df[df['strtaxonomy'] == str_1]['strNode'].value_counts().rename_axis('strNode').reset_index(name='count')
        new = new.head(top)
        class_1 = list(new['strNode'])

        df_class_1 = pd.DataFrame()
        for i in class_1:
            temp = df[df['strNode'] == i]
            df_class_1 = df_class_1.append(temp)

        merge = pd.merge(df_class_1, df[df['strtaxonomy'] == str_1], on=list(df.columns), how='outer', indicator=True)
        others_class_1 = merge[merge['_merge'] == 'right_only']
        others_class_1['strNode'] = 'others'
        final_class_1 = pd.DataFrame()
        final_class_1 = final_class_1.append(merge[merge['_merge'] == 'both'])
        final_class_1 = final_class_1.append(others_class_1)
        
        del final_class_1['_merge']
        
        return final_class_1

    
    def logistic_multiclass_model(self, param_range):
        
        """ self, represents the instance of the class.
            
            Hyperparameter Tuning using GridSearchCV (3-Cross Validation) on Logistic Regression and saves hyperparameter tuned model of Logistic Regression.
            Hyperparameter tuning 'C', 'penality'.
            
            Input
            1. list of various numbers for Hyperparameter 'C'
            
            Output
            1. No Returns, only saves the best model of Logistic Regression in self.path
            
            
            Note: Before you call this function create an object and then call this function
        """

        tuned_parameters = [{'C': param_range, 'penalty':['l1','l2']}]
        model = GridSearchCV(LogisticRegression(class_weight='balanced', random_state=93, multi_class='auto'), tuned_parameters, scoring='f1_micro', cv=3)
        model.fit(self.X_train, self.y_train)
        pred = model.predict(self.X_test)
        print(model.best_estimator_)
        print(model.score(self.X_test, self.y_test))
        print('\n\n')
        
        filename = 'logistic_' + str(self.name) + '.sav'
        pickle.dump(model, open(self.path + filename, 'wb'))
    
    
    def sgd_multiclass_model(self, param_range):
        
        """ self, represents the instance of the class.
            
            Hyperparameter Tuning using GridSearchCV (3-Cross Validation) on SGD Classifier and saves hyperparameter tuned model of SGD Classifier.
            Hyperparameter tuning 'alpha', 'penality'.
            
            Input
            1. list of various numbers for Hyperparameter 'alpha'.
            
            Output
            1. No Returns, only saves the best model of SGD Classifier in self.path
            
            
            Note: Before you call this function create an object and then call this function
        """

        tuned_parameters = [{'alpha': param_range, 'penalty':['l1','l2']}]
        model = GridSearchCV(SGDClassifier(class_weight='balanced', random_state=93), tuned_parameters, scoring='f1_micro', cv=3)
        model.fit(self.X_train, self.y_train)
        pred = model.predict(self.X_test)
        print(model.best_estimator_)
        print(model.score(self.X_test, self.y_test))
        print('\n\n')
        
        filename = 'sgd_' + str(self.name) + '.sav'
        pickle.dump(model, open(self.path + filename, 'wb'))
        
        
    def rfdt_multiclass_model(self, param_range):
        
        """ self, represents the instance of the class.
            
            Hyperparameter Tuning using GridSearchCV (3-Cross Validation) on RandomForestClassifier and saves hyperparameter tuned model of RandomForestClassifier.
            Hyperparameter tuning 'alpha', 'penality'.
            
            Input
            1. list of various numbers for Hyperparameter 'n_estimators'.
            
            Output
            1. No Returns, only saves the best model of RandomForestClassifier in self.path
            
            
            Note: Before you call this function create an object and then call this function
        """

        tuned_parameters = [{'n_estimators': param_range}]
        model = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=93), tuned_parameters, scoring='f1_micro', cv=3)
        model.fit(self.X_train, self.y_train)
        pred = model.predict(self.X_test)
        print(model.best_estimator_)
        print(model.score(self.X_test, self.y_test))
        print('\n\n')
        
        filename = 'rfdt_' + str(self.name) + '.sav'
        pickle.dump(model, open(self.path + filename, 'wb'))
        

    def gbdt_multiclass_model(self, param_range): # work in progress
        
        """ self, represents the instance of the class.
            
            Hyperparameter Tuning using GridSearchCV (3-Cross Validation) on GradientBoostingClassifier and saves hyperparameter tuned model of GradientBoostingClassifier.
            Hyperparameter tuning 'alpha', 'penality'.
            
            Input
            1. list of various numbers for Hyperparameter 'n_estimators'.
            
            Output
            1. No Returns, only saves the best model of GradientBoostingClassifier in self.path
            
            
            Note: Before you call this function create an object and then call this function
        """

        tuned_parameters = [{'n_estimators': param_range}]
        model = GridSearchCV(GradientBoostingClassifier(random_state=93), tuned_parameters, scoring='f1_micro', cv=3)
        model.fit(self.X_train, self.y_train)
        pred = model.predict(self.X_test)
        print(model.best_estimator_)
        print(model.score(self.X_test, self.y_test))
        print('\n\n')
        
        filename = 'gbdt_' + str(self.name) + '.sav'
        pickle.dump(model, open(self.path + filename, 'wb'))
        
        
    def train_test_plot(self, param_range, param_name):
        
        """ self, represents the instance of the class.
        
            Train-Test Plot 
        
            Input
            1. param_range 
                i.e. for hyperparameter 'C' in LR.
                     for hyperparameter 'alpha' in SGD.
            
            Output
            1. No returns, Plots the Train-Test f1_micro plots and also prints Train-Test f1 scores for each Hyperparameter i.e. either 'C' or 'alpha'.
        """
        
        train_scores, test_scores = validation_curve(LogisticRegression(class_weight='balanced', random_state=93, multi_class='multinomial'), self.X_train, self.y_train, 
                                             param_name=param_name, param_range=param_range, scoring='f1_micro', cv=3, n_jobs=1)
              
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        plt.plot(param_range, train_scores_mean, label = "Training Score")
        plt.plot(param_range, test_scores_mean, label = "Test Score")
        plt.grid()
        plt.title("Train-Test Scores Curve for LR with varying '{}' ".format(param_name))
        plt.xlabel("Varying " + param_name + " ----->")
        plt.ylabel("f1_micro Scores ----->")
        plt.legend()
        plt.show()
        print(train_scores_mean)
        print(test_scores_mean)
        print('\n\n')
        
        
    def train_model_scores(self, model_type):
        
        """ self, represents the instance of the class.
        
            Print Train Scores, i.e. calculating 
            1. Precision
            2. Recall
            3. F1 Micro Score
            
            Using Direct functions of sklearn.
        """
    
        loaded_model = pickle.load(open(self.path + str(model_type) + '_' + self.name + '.sav', 'rb'))
        pred = loaded_model.predict(self.X_train)
        pre_score = precision_score(self.y_train, pred, average=None, pos_label=1, sample_weight=None)
        recall = recall_score(self.y_train, pred, average=None)
        f1 = f1_score(self.y_train, pred, average=None)
        print('Train Scores')
        print('Precision =' ,pre_score)
        print('Recall    =' ,recall)
        print('F1 Score  =' ,f1)
        print('\n\n')
        
        
    def test_model_scores(self, model_type):
        
        """ self, represents the instance of the class.
        
            Print Test Scores, i.e. calculating 
            1. Precision
            2. Recall
            3. F1 Micro Score
            
            Using Direct functions of sklearn.
        """
    
        loaded_model = pickle.load(open(self.path + str(model_type) + '_' + self.name + '.sav', 'rb'))
        pred = loaded_model.predict(self.X_test)
        pre_score = precision_score(self.y_test, pred, average=None, pos_label=1, sample_weight=None)
        recall = recall_score(self.y_test, pred, average=None)
        f1 = f1_score(self.y_test, pred, average=None)
        print('Test Scores')
        print('Precision =' ,pre_score)
        print('Recall    =' ,recall)
        print('F1 Score  =' ,f1)
        print('\n\n')
        

    def query_predict(self, df, model_name):
        
        """ self, represents the instance of the class.
        
            Predicting the class label of a New Query given its tokenizer and Model 
        
            Inputs
            1. Basic Pre-processed DataFrame
            2. model_name
                eg: For Logistic Regression, model_name = 'logistic'
                    For SGDClassofoer, model_name = 'sgd'
                    
            Outputs
            1. DataFrame with predicted label  
        """
        
        with open(self.path + 'tfidf_vectorizer_' + self.name + '.pickle', 'rb') as handle:
            count_vec = pickle.load(handle)
        
        X_test = count_vec.transform(df['FOI_TEXT_TRIM'])
        loaded_model = pickle.load(open(self.path + str(model_name) + '_' + self.name + '.sav', 'rb'))
        pred = loaded_model.predict(X_test)
        df['pred'] = pred
        
        return df