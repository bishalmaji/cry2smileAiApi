import os
import pandas as pd
import librosa
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import pickle
import os
import matplotlib.pyplot as plt
os.chdir('D:\Baby Cry Project\cry2smileAiApi')
from sklearn.model_selection import cross_val_score, KFold




# Function to load audio features from the specified directory
def load_audio_features(root_dir):
    chroma_list = []
    spectral_centroid_list = []
    spectral_contrast_list = []
    mfcc_list = []
    zero_crossing_rate_list = []
    cry_category_list = []

    category_mapping = {
        'hungry': 1,
        'burping': 2,
        'belly_pain': 3,
        'discomfort': 4,
        'tired': 5
    }

    for category_name, category_label in category_mapping.items():
        category_dir = os.path.join(root_dir, 'audios', category_name)

        for audio_file in os.listdir(category_dir):
            audio_path = os.path.join(category_dir, audio_file)
            audio_data, sample_rate = librosa.load(audio_path)
            mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_data)

            chroma_list.append(chroma)
            spectral_centroid_list.append(spectral_centroid)
            spectral_contrast_list.append(spectral_contrast)
            mfcc_list.append(mfcc)
            zero_crossing_rate_list.append(zero_crossing_rate)
            cry_category_list.append(category_label)

    data = pd.DataFrame({
        'Chroma': chroma_list,
        'MFCC': mfcc_list,
        'SpectralCentroid': spectral_centroid_list,
        'SpectralContrast': spectral_contrast_list,
        'ZeroCrossingRate': zero_crossing_rate_list,
        'CryCategory': cry_category_list
    })

    return data

# Function to preprocess the loaded data
def preprocess_data(data):
    data['Chroma'] = data['Chroma'].apply(lambda x: [item for sublist in x for item in sublist])
    data['SpectralCentroid'] = data['SpectralCentroid'].apply(lambda x: [item for sublist in x for item in sublist])
    data['SpectralContrast'] = data['SpectralContrast'].apply(lambda x: [item for sublist in x for item in sublist])
    data['MFCC'] = data['MFCC'].apply(lambda x: [item for sublist in x for item in sublist])
    data['ZeroCrossingRate'] = data['ZeroCrossingRate'].apply(lambda x: [item for sublist in x for item in sublist])
    return data

# Function to apply Min-Max scaling
def apply_minmax_scaling(data):
    scaler = MinMaxScaler()
    numeric_columns = data.select_dtypes(include=['int64', 'float64'])
    for column in numeric_columns.columns.difference(['CryCategory']):
        data[column] = scaler.fit_transform(data[[column]])
    return data

# Function to train a Decision Tree classifier
def train_decision_tree_classifier(X, y):
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X, y)
    return decision_tree

# Function to train an SVM classifier
def train_svm_classifier(X, y):
    svm_classifier = SVC(random_state=42)
    svm_classifier.fit(X, y)
    return svm_classifier

# Function to train a Random Forest classifier
def train_randomForest_classifier(X, y):
    random_classifier = RandomForestClassifier(random_state=42)
    random_classifier.fit(X, y)
    return random_classifier

# Main function
def main():
    root_dir = 'D:\\Baby Cry Project\\cry2smileAiApi'
    data = load_audio_features(root_dir)
    data.to_csv('D:\\Baby Cry Project\\baby_cry_dataset.csv', index=False)
    data = preprocess_data(data)

    # Calculate the mean for each feature
    data['MFCC_Mean'] = data['MFCC'].apply(lambda x: np.mean(x))
    data['Chroma_Mean'] = data['Chroma'].apply(lambda x: np.mean(x))
    data['SpectralCentroid_Mean'] = data['SpectralCentroid'].apply(lambda x: np.mean(x))
    data['SpectralContrast_Mean'] = data['SpectralContrast'].apply(lambda x: np.mean(x))
    data['ZeroCrossingRate_Mean'] = data['ZeroCrossingRate'].apply(lambda x: np.mean(x))

    # Apply Min-Max scaling
    data = apply_minmax_scaling(data)
    # Define the columns you want to select
    selected_features = [ 'ZeroCrossingRate_Mean','Chroma_Mean','SpectralCentroid_Mean']

    # Save the scaled dataset to a CSV file
    data.to_csv('D:\\Baby Cry Project\\scaled_baby_cry_dataset.csv', index=False)
    if all(feature in data.columns for feature in selected_features):
        # Train Decision Tree Classifier
        X = data[selected_features]
        y = data['CryCategory']
        decision_tree = train_decision_tree_classifier(X, y)
        # Save the Decision Tree model using pickle
        pickle.dump(decision_tree, open('decision_tree_model.pkl', 'wb'))

        # Train SVM Classifier
        X = data[selected_features]
        y = data['CryCategory']
        svm_classifier = train_svm_classifier(X, y)
        # Save the SVM model using pickle
        pickle.dump(svm_classifier, open('svm_classifier_model.pkl', 'wb'))

        # Train Random Forest Classifier
        selected_features = [ 'ZeroCrossingRate_Mean','SpectralContrast_Mean','SpectralCentroid_Mean']
        X = data[selected_features]
        y = data['CryCategory']
        random_forest_classifier = train_randomForest_classifier(X, y)
        pickle.dump(svm_classifier, open('rfc_model.pkl', 'wb'))


        # Perform k-fold cross-validation (5-fold in this example)
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(random_forest_classifier, X, y, cv=kfold)

        # Calculate and print the accuracy of each fold
        for fold, score in enumerate(scores, 1):
            print(f'Fold {fold} Accuracy: {score:.2f}')

        # Calculate and print the mean accuracy and standard deviation
        mean_accuracy = scores.mean()
        std_accuracy = scores.std()
        print(f'Svm forest Mean Accuracy : {mean_accuracy:.2f}')
        print(f'SVm forest Standard Deviation: {std_accuracy:.2f}')
   
        # Save the Decision Tree model using pickle
        with open('decision_tree_model.pkl', 'wb') as decision_tree_file:
            pickle.dump(decision_tree, decision_tree_file)

        # Save the SVM model using pickle
        with open('svm_classifier_model.pkl', 'wb') as svm_file:
            pickle.dump(svm_classifier, svm_file)

        # Save the Random Forest model using pickle
        with open('rfc_model.pkl', 'wb') as random_forest_file:
            pickle.dump(random_forest_classifier, random_forest_file)

        print("Models trained successfully.")
    else:
        print("Selected feature columns not found in the DataFrame.")

if __name__ == '__main__':
    main()
