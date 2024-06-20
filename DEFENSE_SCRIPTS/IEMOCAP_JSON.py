import json
import pandas as pd

# Function to load JSON data
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to extract filename and emotion and save to a DataFrame
def extract_data_to_dataframe(json_data):
    # Initialize a list to store the extracted data
    extracted_data = []
    
    # Iterate through each item in the JSON data
    for key, value in json_data.items():
        filename = key
        emotion = value['emo']
        extracted_data.append({'filename': filename, 'Emotion': emotion})
    
    # Create a DataFrame
    df = pd.DataFrame(extracted_data)
    return df

# Example usage
if __name__ == '__main__':
    file_path = r"C:\MyDocs\DTU\MSc\Thesis\github\low-resource-emotion-recognition\DEFENSE_SCRIPTS\train.json"  # Replace this with the path to your JSON file
    json_data = load_json(file_path)
    result_df = extract_data_to_dataframe(json_data)
    print(result_df.head(20))

    # print shape of result_df
    print(result_df.shape)

    # save result_df to csv
    result_df.to_csv(r"C:\MyDocs\DTU\MSc\Thesis\github\low-resource-emotion-recognition\DEFENSE_SCRIPTS\IEMOCAP.csv", index=False)
