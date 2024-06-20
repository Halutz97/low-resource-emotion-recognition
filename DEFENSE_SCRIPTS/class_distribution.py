import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Plot label distribution
def plot_label_distribution(data, column):
    # Count the frequency of each category
    label_counts = data[column].value_counts()

    # Plotting
    plt.figure(figsize=(10, 6))
    label_counts.plot(kind='bar', color='skyblue')
    plt.title('Class dist. IEMOCAP FULL (total = 10039)')
    plt.xlabel('Emotions')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # save figure
    plt.savefig('DEFENSE_SCRIPTS/label_distribution_IEMOCAP_FULL.png')
    plt.show()

# Example usage
if __name__ == '__main__':
    train_data = pd.read_csv(r"C:\MyDocs\DTU\MSc\Thesis\Data\IEMOCAP\IEMOCAP_full_release\labels.csv")
    # results = results.dropna(subset=['audio_prob_class', 'video_prob'])
    
    # with open('multimodal_results/multimodal_baseline_test_set.pkl', 'rb') as f:
        # selected_files = pickle.load(f)

    # test_data = results[results['filename'].isin(selected_files)]

    # train_data = results[~results['filename'].isin(selected_files)]

    # Check shapes
    print("Train data shape:")
    print(train_data.shape)
    print()
    # print("Test data shape:")
    # print(test_data.shape)
    # print()

    # file_path = 'path_to_your_csv.csv'  # Update this to the path of your CSV file
    # data = load_data(file_path)
    plot_label_distribution(train_data, 'Emotion')  # Update 'emotion_column' to the name of your emotion label column
