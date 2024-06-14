# Description: This script generates an overview of all results to be reported.

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics
from scipy import stats

def argmax_str_to_array(argmax_str):
    argmax_str = argmax_str[1:-1]
    argmax_str = argmax_str.split()
    argmax = [int(i) for i in argmax_str]
    return argmax

def create_confusion_matrix(data, use_voted_labels=True, show_confusion_matrix=True):
    label_model = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']
    encoding_dict_dataset = {'Neutral': 0, 'Happiness': 1, 'Sadness': 2, 'Surprise': 3, 'Fear': 4, 'Disgust': 5, 'Anger': 6}
    label_names = label_model
    if use_voted_labels:
        true_labels = data['emotion_label_voted']
    else:
        true_labels = data['emotion_label_self_reported']

    label_keys = [encoding_dict_dataset[label] for label in true_labels]

    predicted_classes= data['predicted']
    
    predicted_keys = [encoding_dict_dataset[label] for label in predicted_classes]

    # print(predicted_keys == label_keys)

    # Convert predicted_keys and label_keys to numpy arrays
    predicted_keys = np.array(predicted_keys)
    label_keys = np.array(label_keys)

    # # Calculate accuracy
    accuracy = (predicted_keys == label_keys).mean()
    print("Accuracy:", accuracy)

    # # Calculate F1 Scores
    f1_micro = sklearn.metrics.f1_score(label_keys, predicted_keys, average='micro')
    print("F1 Score (Micro):", f1_micro)

    f1_macro = sklearn.metrics.f1_score(label_keys, predicted_keys, average='macro')
    print("F1 Score (Macro):", f1_macro)

    f1_weighted = sklearn.metrics.f1_score(label_keys, predicted_keys, average='weighted')
    print("F1 Score (Weighted):", f1_weighted)

    # # Generate confusion matrix
    confusion_matrix = sklearn.metrics.confusion_matrix(label_keys, predicted_keys)

    confusion_matrix_full = np.zeros((len(label_names), len(label_names)), dtype=int)

    # # Fill the confusion matrix with the values from the actual confusion matrix
    for i, label in enumerate(true_labels):
        confusion_matrix_full[label_keys[i], predicted_keys[i]] +=1

    # # Create a DataFrame for the confusion matrix
    confusion_matrix_df = pd.DataFrame(confusion_matrix_full, index=label_names, columns=label_names)

    # Normalize each row by its sum to get the percentage
    confusion_matrix_percent = confusion_matrix_df.div(confusion_matrix_df.sum(axis=1), axis=0) * 100

    # Print the shape of confusion_matrix_percent
    print("Shape of confusion_matrix_percent:")
    print(confusion_matrix_percent.shape)

    # What data type are the elements in confusion_matrix_percent?
    print("Data type of elements in confusion_matrix_percent:")
    print(confusion_matrix_percent.dtypes)

    # Round the numbers in confusion_matrix_percent to one decimal
    confusion_matrix_percent = confusion_matrix_percent.round(1)

    # If an element is NaN, replace it with 0
    confusion_matrix_percent = confusion_matrix_percent.fillna(0)

    # Create a DataFrame for the normalized confusion matrix
    confusion_matrix_df_percent = pd.DataFrame(confusion_matrix_percent, index=label_names, columns=label_names)

    # # Add a row and column for the total counts
    confusion_matrix_df['Total'] = confusion_matrix_df.sum(axis=1)
    confusion_matrix_df.loc['Total'] = confusion_matrix_df.sum()

    print("Confusion Matrix:")
    # print(confusion_matrix_df)
    print(confusion_matrix_df_percent)

    # # Calculate the maximum value for the heatmap color scale
    # max_value = confusion_matrix_df.iloc[:-1,:].values.max()
    max_value = 100
    # if show_confusion_matrix:
    #     plt.figure(figsize=(8, 8))
    #     # sns.heatmap(confusion_matrix_df.iloc[:-1, :-1], annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names, vmin=0, vmax=max_value)
    #     sns.heatmap(confusion_matrix_df_percent, annot=True, fmt='.1f', cmap='Blues', xticklabels=label_names, yticklabels=label_names, vmin=0, vmax=max_value)
    #     plt.title('Confusion Matrix')
    #     plt.xlabel('Predicted')
    #     plt.ylabel('Actual')
    #     plt.show()

    # Plotting the confusion matrix
    plt.figure(figsize=(10, 8))  # Adjust figsize to fit your thesis layout
    ax = sns.heatmap(confusion_matrix_df_percent, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=label_names, yticklabels=label_names,
                    vmin=0, vmax=max_value, cbar_kws={'shrink': 0.8})  # Control the size of the color bar
    # plt.title('Confusion Matrix (%)', fontsize=16)  # Title with fontsize
    plt.xlabel('Predicted label', fontsize=14)  # X-axis label with fontsize
    plt.ylabel('Actual label', fontsize=14)  # Y-axis label with fontsize
    plt.xticks(rotation=45)  # Rotate x labels for better fit
    plt.yticks(rotation=0)  # Keep y labels horizontal for readability
    plt.tight_layout()  # Adjust layout to not cut-off labels
    plt.savefig('Ari_figures/confusion_matrix_FER_validation_voted.png', dpi=300, bbox_inches='tight')
    plt.show()
    # Save figure as png
    

def generate_voted_labels(original_dataframe, voted_labels_path):
    print("------------------------------------------------------------")
    print("-----------------generate_voted_labels_data-----------------")
    print("------------------------------------------------------------")
    voted_labels = pd.read_csv(voted_labels_path)
    voted_labels['filename'] = voted_labels['filename'] + '.mp4'
    print()
    print("Voted labels: ")
    print()
    print(voted_labels.head(10))
    voted_labels_files = voted_labels['filename'].tolist()
    filtered_data = original_dataframe[original_dataframe['filename'].isin(voted_labels_files)].copy()
    filtered_data_files = filtered_data['filename'].tolist()
    voted_labels = voted_labels[voted_labels['filename'].isin(filtered_data_files)]
    voted_labels_files = voted_labels['filename'].tolist() # update it
    voted_emotion_dict = {'N': 'Neutral', 'H': 'Happiness', 'S': 'Sadness', 'F': 'Fear', 'D': 'Disgust', 'A': 'Anger'}
    voted_labels = voted_labels.drop(columns=['Level'])
    voted_labels['emotion_label_voted'] = voted_labels['Emotion'].apply(lambda x: voted_emotion_dict[x])
    voted_labels = voted_labels.drop(columns=['Emotion'])
    voted_labels = voted_labels.rename(columns={'filename': 'filename_voted'}) # rename column filename to filename_voted
    filtered_data = filtered_data.rename(columns={'emotion': 'emotion_label_self_reported'}) # rename column filename to filename_voted

    filtered_data_files.sort()
    voted_labels_files.sort()
    if filtered_data_files == voted_labels_files:
        print()
        print("Files filtered successfully.")
        print()
    else:
        print()
        print("Files not filtered successfully.")
        print()
        return None
    # print("Voted labels after processing: ")
    # print()
    # print(voted_labels.head(10))
    # print()
    # print("Original data filtered: ")
    # print()
    # print(filtered_data.head(10))
    filtered_data = filtered_data.sort_values(by='filename').reset_index(drop=True)
    voted_labels = voted_labels.sort_values(by='filename_voted').reset_index(drop=True)
    predictions_voted_labels = pd.concat([filtered_data, voted_labels], axis=1) # concatenate horizontally
    # print()
    # print("Predictions with self-reported labels and voted labels concatenated:")
    # print()
    # print(predictions_voted_labels.head(10))
    predictions_voted_labels = predictions_voted_labels.drop(columns=['filename_voted'])
    predictions_voted_labels = predictions_voted_labels[['filename', 'emotion_label_self_reported', 'emotion_label_voted', 'predicted', 'argmax']]
    # predictions_voted_labels = predictions_voted_labels.drop(columns=['emotion_self_reported_label'])
    # predictions_voted_labels = predictions_voted_labels.rename(columns={'emotion_label': 'emotion'}) # rename column emotion_label to emotion
    print()
    print("Final dataframe with self reported and voted labels:")
    print()
    print(predictions_voted_labels.head(10))
    return predictions_voted_labels

def main():
    # Load CSV file(s) with true and predicted labels
    data1 = pd.read_csv(r'ryumina_fer_model\save_results\run_5_predicted_checkpoint_1150.csv')
    data2 = pd.read_csv(r'ryumina_fer_model\save_results\run_5_predicted_checkpoint_2150.csv')
    data3 = pd.read_csv(r'ryumina_fer_model\save_results\run_5_predicted_checkpoint_6150.csv')
    data4 = pd.read_csv(r'ryumina_fer_model\save_results\run_5_predicted_checkpoint_7442.csv')

    # data1 = pd.read_csv(r'save_results\run_4_predicted_checkpoint_1000.csv')
    # data2 = pd.read_csv(r'save_results\run_4_predicted_checkpoint_4400.csv')


    # Concatenate vertically
    data = pd.concat([data1, data2, data3, data4], axis=0)
    # data = pd.concat([data1, data2], axis=0)

    # get the shape of the dataframe
    print(data.shape)

    # remove the rows with missing values
    # First check if there are any missing values
    print(data.isnull().sum())
    # Print the rows with missing values
    print(data[data.isnull().any(axis=1)])

    # Drop the rows with missing values
    data = data.dropna()
    # Print the new shape of the dataframe
    print(data.shape)

    # print(data.head(10))

    voted_labels_path = r"C:\MyDocs\DTU\MSc\Thesis\Data\CREMA-D\CREMA-D\voted_face_labels.csv"
    full_data = generate_voted_labels(data, voted_labels_path)
    # create_confusion_matrix(full_data, use_voted_labels=False, show_confusion_matrix=True)
    create_confusion_matrix(full_data, use_voted_labels=True, show_confusion_matrix=True)

    


    # ## Investigate if we only take files ending with "HI.mp4"
    # # Read the 'argmax' column from the first row
    # file = data.iloc[104]['filename']
    # print(file)

    # # Using boolean indexing, create a list of dataframe rows from data that meet the condition
    # # data_filtered = data[data['filename'].str.endswith('XX.mp4')]

    # # Using boolean indexing, create a list of dataframe rows from data that does not meet the condition
    # data_filtered = data[~data['filename'].str.endswith('LO.mp4')]
    # data_filtered = data_filtered[~data_filtered['filename'].str.endswith('MD.mp4')]


    # # data_filtered = data.copy()

    # print(data_filtered.shape)
    # print(data_filtered.head(10))
    # # print(argmax)
    # # argmax = argmax_str_to_array(argmax)
    # # print(argmax)
    # # print(type(argmax[0]))

    # # # Find the unique numbers in argmax
    # # unique_numbers = np.unique(argmax)
    # # print(unique_numbers)

    # # save data_filtered to a csv file
    # # data_filtered.to_csv('data_filtered.csv', index=False)

    # label_model = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']

    # # Create a new column, with argmax converted to array
    # data_filtered['argmax_array'] = data_filtered['argmax'].apply(argmax_str_to_array)
    # # print(data_filtered.head(10))
    # # Create new column with unique values in argmax_array
    # data_filtered['unique_values'] = data_filtered['argmax_array'].apply(np.unique)
    # # print(data_filtered.head(20))
    # # Create new column number_unique with the number of unique values in argmax_array
    # data_filtered['number_unique'] = data_filtered['unique_values'].apply(len)
    # # print(data_filtered.head(10))
    # # Create new column with true if number_unique is 2 and one of the numbers is 0, false otherwise
    # data_filtered['two_unique'] = (data_filtered['number_unique'] == 2) & (data_filtered['unique_values'].apply(lambda x: 0 in x))
    # # data_filtered['two_unique'] = data_filtered['number_unique'] == 2
    # # Drop the checkpoint column and argmax_array column
    # data_filtered = data_filtered.drop(columns=['checkpoint', 'argmax_array'])
    # # Create new columns with the max value in unique_values
    # data_filtered['max_value'] = data_filtered['unique_values'].apply(np.max)
    # # Create new column with label model applied to "max_value"
    # data_filtered['new_prediction'] = data_filtered['max_value'].apply(lambda x: label_model[x])
    # # Create new column "prediction_corrected" equal to "new_prediction" if "two_unique" is true, otherwise equal to "predicted"
    # data_filtered['prediction_corrected'] = data_filtered.apply(lambda x: x['new_prediction'] if x['two_unique'] else x['predicted'], axis=1)
    # # drop columns "number_unique", "two_unique", "new_prediction", "max_value"
    # data_filtered = data_filtered.drop(columns=['number_unique', 'two_unique', 'new_prediction', 'max_value'])
    # # Drop the column predicted
    # data_filtered = data_filtered.drop(columns=['predicted'])
    # # Rename the column "prediction_corrected" to "predicted"
    # data_filtered = data_filtered.rename(columns={'prediction_corrected': 'predicted'})
    # # reset indexing
    # data_filtered = data_filtered.reset_index(drop=True)
    # print(data_filtered.head(20))

    # create_confusion_matrix(data_filtered, show_confusion_matrix=True)

if __name__ == '__main__':
    main()