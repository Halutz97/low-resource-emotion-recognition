import pandas as pd
import os

def match_emotion_labels(labels_file, corrected_labels_file, directory, dataset="MELD", attributes=False):
    # Assuming you have a DataFrame with columns "filename" and "emotion"
    
    data = pd.read_csv(labels_file)

    # Iterate through dataframe:
    if dataset == "MELD":
        for index, row in data.iterrows():
            dialogue_id = str(row['Dialogue_ID'])
            utterance_id = str(row['Utterance_ID'])
            if int(dialogue_id) >= 1000:
                    dialogue_id = dialogue_id
            elif int(dialogue_id) >= 100:
                dialogue_id = "0" + dialogue_id
            elif int(dialogue_id) >= 10:
                dialogue_id = "00" + dialogue_id
            else:
                dialogue_id = "000" + dialogue_id

            if int(utterance_id) >= 10:
                utterance_id = utterance_id
            else:
                utterance_id = "0" + utterance_id

            row['Expected filename'] = "dia" + dialogue_id + "_utt" + utterance_id + ".wav"
            data.at[index, 'Expected filename'] = row['Expected filename']

    files = []

    # Get a list of all files in the directory
    for file in os.listdir(directory):
        if file.endswith('.wav'):
            files.append(file)

    # sort 'files' alphabetically
    files.sort()

    # Print number of files in directory
    print()
    print("Number of audio files in directory: " + str(len(files)))
    print()

    # Print lenght of dataframe
    print("Number of entries in dataframe: " + str(len(data)))
    print()


    if dataset == "MELD":
        # store a list of files that do not have labels
        files_without_labels = files.copy()
        # iterate through the column 'Expected filename' and check if any filenames are not in 'files'
        for index, row in data.iterrows():
            if row['Expected filename'] in files:
                # remove "expected filename" from 'files'
                files_without_labels.remove(row['Expected filename'])
            
            # if row['Expected filename'] not in files:
                # files_without_labels.append(row['Expected filename'])

        # print()
        print("Files without labels:")
        print(files_without_labels)
        print()

        # remove files that do not have labels from 'files' and delete them from the directory
        deleted_files = 0
        for file in files_without_labels:
            # delete audio file drom directory
            os.remove(os.path.join(directory, file))
            deleted_files += 1
            files.remove(file)
        # print()
        print("Deleted files: " + str(deleted_files))
        print()

        # print(files)
        # print()

        # Check if there are any duplicate files in 'files'
        # duplicates = set([x for x in files if files.count(x) > 1])
        # print("Duplicates:")
        # print(duplicates)
        # print()


        # lenght of data BEFORE
        # print()
        print("Length of data BEFORE:")
        print(len(data))
        print()

        # remove dataframe rows if file is not in 'files'
        print("Removing rows from dataframe if file is not in 'files'...")
        print()
        data = data[data['Expected filename'].isin(files)]

        # lenght of data AFTER
        # print()
        print("Length of data AFTER:")
        print(len(data))
        print()

        files.sort()

        # Add filenames to a new column in the DataFrame
        data['filename'] = files

        # Iterate through dataframe and check if any filenames do not match the expected filenames
        num_missmatches = 0
        for index, row in data.iterrows():
            if row['filename'] != row['Expected filename']:
                print(row['filename'])
                print(row['Expected filename'])
                print()
                num_missmatches += 1

        # print()
        print("Number of missmatches: " + str(num_missmatches))
        print()

    elif dataset == "EMO-DB":
        # Drop all rows with oth, xxx, dis, and fea of the csv file
        data = data[data['Emotion'] != 'L']
        data = data[data['Emotion'] != 'E']
        data = data[data['Emotion'] != 'A']


        print("Number of entries in dataframe after removing some emotions: " + str(len(data)))

        # Eliminate all wav files from directory that do not appear on data
        num_deleted_files = 0
        for file in files:
            if file not in data['filename'].values:
                print("File not in data: " + file)
                num_deleted_files += 1
                os.remove(os.path.join(directory, file))

        print("Number of deleted files: " + str(num_deleted_files))
        print("Number of files in directory after deletion: " + str(len(os.listdir(directory))))

        # Eliminate all rows from data that do not have a corresponding wav file
        data = data[(data['filename']).isin(files)]
        print("Number of entries in dataframe after removing files not in directory: " + str(len(data)))

    elif dataset == "ShEMO":
        # Drop all rows with oth, xxx, dis, and fea of the csv file
        data = data[data['Emotion'] != 'W']
        data = data[data['Emotion'] != 'F']


        print("Number of entries in dataframe after removing some emotions: " + str(len(data)))

        # Eliminate all wav files from directory that do not appear on data
        num_deleted_files = 0
        for file in files:
            if file not in data['filename'].values:
                print("File not in data: " + file)
                num_deleted_files += 1
                os.remove(os.path.join(directory, file))

        print("Number of deleted files: " + str(num_deleted_files))
        print("Number of files in directory after deletion: " + str(len(os.listdir(directory))))

        # Eliminate all rows from data that do not have a corresponding wav file
        data = data[(data['filename']).isin(files)]
        print("Number of entries in dataframe after removing files not in directory: " + str(len(data)))

    elif dataset == "CREMA-D":
        # Drop all rows with oth, xxx, dis, and fea of the csv file
        data = data[data['Emotion'] != 'DIS']
        data = data[data['Emotion'] != 'FEA']




        print("Number of entries in dataframe after removing some emotions: " + str(len(data)))

        # Eliminate all wav files from directory that do not appear on data
        num_deleted_files = 0
        for file in files:
            if file not in data['filename'].values:
                print("File not in data: " + file)
                num_deleted_files += 1
                os.remove(os.path.join(directory, file))

        print("Number of deleted files: " + str(num_deleted_files))
        print("Number of files in directory after deletion: " + str(len(os.listdir(directory))))

        # Eliminate all rows from data that do not have a corresponding wav file
        data = data[(data['filename']).isin(files)]
        print("Number of entries in dataframe after removing files not in directory: " + str(len(data)))

    elif dataset == "CREMA-D-voted":
        # Drop all rows with oth, xxx, dis, and fea of the csv file
        data = data[data['Emotion'] != 'D']
        data = data[data['Emotion'] != 'F']


        print("Number of entries in dataframe after removing some emotions: " + str(len(data)))

        # Eliminate all wav files from directory that do not appear on data
        num_deleted_files = 0
        for file in files:
            if file not in (data['filename'].values+'.wav'):
                print("File not in data: " + file)
                num_deleted_files += 1
                os.remove(os.path.join(directory, file))

        print("Number of deleted files: " + str(num_deleted_files))
        print("Number of files in directory after deletion: " + str(len(os.listdir(directory))))

        # Eliminate all rows from data that do not have a corresponding wav file
        data = data[(data['filename']+'.wav').isin(files)]
        print("Number of entries in dataframe after removing files not in directory: " + str(len(data)))

    elif dataset == "IEMOCAP" and attributes == False:
        # Drop all rows with oth, xxx, dis, and fea of the csv file
        data = data[data['Emotion'] != 'oth']
        data = data[data['Emotion'] != 'xxx']
        data = data[data['Emotion'] != 'dis']
        data = data[data['Emotion'] != 'fea']

        data = data[data['Emotion'] != 'sur']
        data = data[data['Emotion'] != 'fru']

        data.loc[data['Emotion'] == 'exc', 'Emotion'] = 'hap'


    if dataset == "IEMOCAP":

        # Separete attributes and drop '[', ']', and ' '
        # data['Attributes'] = data['Attributes'].str.replace('"', '')

        data['Valence'] = data['Attributes'].apply(lambda x: x.split(',')[0])
        data['Arousal'] = data['Attributes'].apply(lambda x: x.split(',')[1])
        data['Dominance'] = data['Attributes'].apply(lambda x: x.split(',')[2])

        data['Valence'] = data['Valence'].str.replace('[', '')
        data['Arousal'] = data['Arousal'].str.replace(' ', '')
        data['Dominance'] = data['Dominance'].str.replace(']', '')
        data['Dominance'] = data['Dominance'].str.replace(' ', '')
        data['Dominance'] = data['Dominance'].str.replace('"', '')




        print("Number of entries in dataframe after removing some emotions: " + str(len(data)))

        # Eliminate all wav files from directory that do not appear on data
        num_deleted_files = 0
        for file in files:
            if file[:-4] not in data['filename'].values:
                print("File not in data: " + file)
                num_deleted_files += 1
                os.remove(os.path.join(directory, file))

        print("Number of deleted files: " + str(num_deleted_files))
        print("Number of files in directory after deletion: " + str(len(os.listdir(directory))))

        # Eliminate all rows from data that do not have a corresponding wav file
        data = data[(data['filename']+'.wav').isin(files)]
        print("Number of entries in dataframe after removing files not in directory: " + str(len(data)))
    


    # Placeholder for labels
    labels = []

    if dataset == "MELD":
        my_encoding_dict = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6}
    elif dataset == "CREMA-D":
        # my_encoding_dict = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
        my_encoding_dict = {'ANG': 0, 'NEU': 1, 'HAP': 2, 'SAD': 3}
    elif dataset == "CREMA-D-voted":
        my_encoding_dict = {'A': 0, 'N': 1, 'H': 2, 'S': 3}
    elif dataset == "EMO-DB":
        my_encoding_dict = {'W': 0, 'N': 1, 'F': 2, 'T': 3}
    elif dataset == "ShEMO":
        my_encoding_dict = {'A': 0, 'N': 1, 'H': 2, 'S': 3}
    elif dataset == "IEMOCAP" and attributes == True:
        my_encoding_dict = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3, 'sur': 4, 'fru': 5, 'exc': 6, 'oth': 7, 'xxx': 8, 'dis': 9, 'fea': 10}
    elif dataset == "IEMOCAP" and attributes == False:
        my_encoding_dict = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3}
    
    
    labels = data['Emotion'].map(my_encoding_dict).values


    print(labels)
    print()
    print(my_encoding_dict)
    print()

    # iterate through dataframe and check if encoding is correct

    # length of dataframe
    print("Length of dataframe (csv):")
    print(len(data))
    print()

    # length of labels
    print("Length of labels:")
    print(len(labels))
    print()

    # reset index of dataframe
    data.reset_index(drop=True, inplace=True)

    num_missmatches = 0
    for index, row in data.iterrows():
        if my_encoding_dict[row['Emotion']] != labels[index]:
                print("Label missmatch:")
                print(row['Emotion'])
                print("Encoding: " + str(labels[index]))
                print("Intended encoding: " + str(my_encoding_dict[row['Emotion']]))
                print()
                num_missmatches += 1

    print("Number of missmatches (label vs encoding): " + str(num_missmatches))
    print()

    # Add labels to dataframe
    data['Label'] = labels

    # ========================================================
    # Export dataframe to csv
    # data.to_csv(corrected_labels_file, index=False)
    # ========================================================
    # df_check = pd.read_csv(r"C:\MyDocs\DTU\MSc\Thesis\Data\MELD\MELD_dataset\train\train_labels_corrected.csv")
    # print()
    # print("df_check:")

    # print(df_check.head())

    # print()

    # Drop columns Sr No., Utterance, Speaker, Sentiment, Dialogue_ID, Utterance_ID, Season, Episode, StartTime, EndTime, Expected filename, Match
    # data = data.drop(['Sr No.', 'Utterance', 'Speaker', 'Sentiment', 'Dialogue_ID', 'Utterance_ID', 'Season', 'Episode', 'StartTime', 'EndTime', 'Expected filename', 'Match'], axis=1)
    if dataset == "MELD":
        data = data.drop(['Sr No.', 'Utterance', 'Speaker', 'Sentiment', 'Dialogue_ID', 'Utterance_ID', 'Season', 'Episode', 'StartTime', 'EndTime', 'Expected filename'], axis=1)
    elif dataset == "CREMA-D":
        data = data.drop(['Index', 'Speaker', 'Line', 'Intensity'], axis=1)
    elif dataset == "CREMA-D-voted":
        data = data.drop(['Level'], axis=1)
    elif dataset == "EMO-DB":
        data = data.drop(['Speaker', 'Line', 'Version'], axis=1)
    elif dataset == "ShEMO":
        data = data.drop(['Speaker', 'Gender', 'Number'], axis=1)
    elif dataset == "IEMOCAP":
        data = data.drop(['Time'], axis=1)

    # data.reset_index(drop=True, inplace=True)

    # Change order of columns and drop unnecessary columns
    if dataset == "IEMOCAP":
        data = data[['filename', 'Emotion', 'Label', 'Valence', 'Arousal', 'Dominance']]
    else:
        data = data[['filename', 'Emotion', 'Label']]


    data.to_csv(corrected_labels_file, index=False)


if __name__ == "__main__":
    labels_file = r"C:\MyDocs\DTU\MSc\Thesis\Data\MELD\Automation_testing\dev_sent_emo.csv"
    corrected_labels_file = r"C:\MyDocs\DTU\MSc\Thesis\Data\MELD\Automation_testing\dev_sent_emo_corrected2.csv"
    directory = r"C:\MyDocs\DTU\MSc\Thesis\Data\MELD\Automation_testing\videos_audio"
    match_emotion_labels(labels_file, corrected_labels_file, directory)