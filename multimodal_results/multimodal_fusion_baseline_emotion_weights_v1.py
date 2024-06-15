# This script reads and processes the results of the multimodal experiments
# Furthermore, a logic for determining the best model is implemented
# That is, automatically adjusting the audio and visual weights

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import pickle
import sklearn.metrics
import seaborn as sns
import matplotlib.pyplot as plt


def create_confusion_matrix(data):
    
    label_model = ['Neutral', 'Anger', 'Happiness', 'Sadness', 'Fear', 'Disgust']
    label_names = label_model

    true_labels = [label_model[label] for label in data['Label']]

    label_keys = data['Label']

    predicted_keys = data['combined_predicted']
    # predicted_keys = data['audio_predicted_single']
    # predicted_keys = data['video_predicted_single']

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
    confusion_matrix_full = np.zeros((len(label_names), len(label_names)), dtype=int)

    # # Fill the confusion matrix with the values from the actual confusion matrix
    for i, label in enumerate(true_labels):
        confusion_matrix_full[label_keys[i], predicted_keys[i]] +=1

    # # Create a DataFrame for the confusion matrix
    confusion_matrix_df = pd.DataFrame(confusion_matrix_full, index=label_names, columns=label_names)

    # Normalize each row by its sum to get the percentage
    confusion_matrix_percent = confusion_matrix_df.div(confusion_matrix_df.sum(axis=1), axis=0) * 100

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
    plt.savefig('Ari_figures/multimodal_baseline_cross_entropy.png', dpi=300, bbox_inches='tight')
    plt.show()
    # Save figure as png

def convert_strings_to_arrays(list_of_strings, pad_to_length=7):
    """
    Convert probabiliteis given as strings to numpy arrays (1,7)
    :param list_of_strings: list of strings
    :return: list of numpy arrays
    """
    list_of_arrays = []
    # i = 0
    for probabilities in list_of_strings:
        probabilities = probabilities.split(", ")
        # if i == 0:
            # print(probabilities)
        probabilities = [x.replace('[', '') for x in probabilities]
        # if i == 0:
            # print(probabilities)
        probabilities = [x.replace(']', '') for x in probabilities]
        # if i == 0:
            # print(probabilities)
        probabilities = [x.replace('\n', '') for x in probabilities]
        # if i == 0:
            # print(probabilities)
        probabilities = [x for x in probabilities if x] # remove empty strings
        # if i == 0:
            # print(probabilities)
        probs_float = [float(x) for x in probabilities]
        # if i == 0:
            # print("Yippie ki yay!")
            # print(probs_float)
        if len(probs_float) < pad_to_length:
            probs_float += [0] * (pad_to_length - len(probs_float))
        # if i == 0:
            # print("Weee padded!")
            # print(probs_float)
        probs_float = np.array(probs_float).reshape(1, pad_to_length)
        # if i == 0:
            # print("Wubba lubba dub dub!")
            # print(probs_float)
        list_of_arrays.append(probs_float)
        # if i == 0:
            # print("Yes")
            # print(list_of_arrays)
        # i += 1
    return list_of_arrays

def evaluate(w_a, w_v, a, v, true_labels):
    # print("-------------------------------------------------")
    # print(f"Evaluating with weights: w_a={w_a}, w_v={w_v}")
    # print("-------------------------------------------------")
    combined_prob = w_a * a + w_v * v
    # print("Combined prob shape: ", combined_prob.shape)
    predictions = np.argmax(combined_prob, axis=1)
    # print("Predictions shape: ", predictions.shape)
    true_classes = np.argmax(true_labels, axis=1)
    # print("True classes shape: ", true_classes.shape)
    # print("predictions == true_classes")
    # print(predictions == true_classes)
    accuracy = np.mean(predictions == true_classes)
    # print(f"Accuracy: {accuracy}")
    return accuracy

def optimize_accuracy(weights, *args):
    # Unpack arguments
    audio_probs, video_probs, true_labels = args

    # Assuming we are optimizing the first four weights for each modality
    w_a = np.concatenate([weights, [0, 0, 0]])  # Last three weights for audio are zero
    # determine w_v as the element-wise difference between 1 and w_a
    w_v = 1 - w_a

    # Calculate the combined probabilities
    combined_probs = w_a * audio_probs + w_v * video_probs

    predictions = np.argmax(combined_probs, axis=1)
    # print("Predictions shape: ", predictions.shape)
    true_classes = np.argmax(true_labels, axis=1)
    # print("True classes shape: ", true_classes.shape)
    # print("predictions == true_classes")
    # print(predictions == true_classes)
    accuracy = np.mean(predictions == true_classes)
    neg_accuracy = -accuracy
    # print(f"Accuracy: {accuracy}")
    return neg_accuracy

def loss_function(weights, *args):
    # Unpack arguments
    audio_probs, video_probs, true_labels = args

    # Assuming we are optimizing the first four weights for each modality
    w_a = np.concatenate([weights, [0, 0, 0]])  # Last three weights for audio are zero
    # determine w_v as the element-wise difference between 1 and w_a
    w_v = 1 - w_a

    # Calculate the combined probabilities
    combined_probs = w_a * audio_probs + w_v * video_probs

    # Example: Mean Squared Error
    mse = np.mean((combined_probs - true_labels) ** 2)
    return mse

def loss_function_CE(weights, *args):
    # Unpack arguments
    audio_probs, video_probs, true_labels = args

    # Assuming we are optimizing the first four weights for each modality
    w_a = np.concatenate([weights, [0, 0]])  # Last three weights for audio are zero
    # determine w_v as the element-wise difference between 1 and w_a
    w_v = 1 - w_a

    # Calculate the combined probabilities
    combined_probs = w_a * audio_probs + w_v * video_probs

    # Ensure numerical stability and avoid log(0) by clipping probabilities
    combined_probs = np.clip(combined_probs, 1e-15, 1 - 1e-15)

    # Cross-entropy loss calculation
    cross_entropy_loss = -np.sum(true_labels * np.log(combined_probs))

    return cross_entropy_loss

def categorical_hinge_loss(weights, *args):
    audio_probs, video_probs, true_labels = args

    w_a = np.concatenate([weights, [0, 0, 0]])
    w_v = 1 - w_a

    combined_probs = w_a * audio_probs + w_v * video_probs

    # Find the maximum predicted probability of incorrect classes
    true_class_probs = np.sum(combined_probs * true_labels, axis=1)
    max_incorrect_class_probs = np.max(combined_probs * (1 - true_labels), axis=1)
    
    # Categorical hinge loss
    loss = np.mean(np.maximum(0, 1 + max_incorrect_class_probs - true_class_probs))
    return loss

def focal_loss(weights, *args, gamma=2):
    audio_probs, video_probs, true_labels = args

    w_a = np.concatenate([weights, [0, 0, 0]])
    w_v = 1 - w_a

    combined_probs = w_a * audio_probs + w_v * video_probs

    # Avoid log(0)
    combined_probs = np.clip(combined_probs, 1e-8, 1 - 1e-8)
    
    # Focal loss calculation
    loss = -np.sum(true_labels * (1 - combined_probs)**gamma * np.log(combined_probs))
    return np.mean(loss)

def kl_divergence_loss(weights, *args):
    audio_probs, video_probs, true_labels = args

    w_a = np.concatenate([weights, [0, 0, 0]])
    w_v = 1 - w_a

    combined_probs = w_a * audio_probs + w_v * video_probs

    # Ensure numerical stability
    combined_probs = np.clip(combined_probs, 1e-15, 1 - 1e-15)
    true_labels = np.clip(true_labels, 1e-15, 1 - 1e-15)

    # KL Divergence
    loss = np.sum(true_labels * np.log(true_labels / combined_probs))
    return np.mean(loss)


def get_single_modality_accuracy(predictions, true_labels):
    predictions = np.argmax(predictions, axis=1)
    true_classes = np.argmax(true_labels, axis=1)
    accuracy = np.mean(predictions == true_classes)
    return accuracy

## ================ Combine results ========================================
# # Read the results
# data1 = pd.read_csv('multimodal_results/run_2_predicted_checkpoint_1440.csv')
# data2 = pd.read_csv('multimodal_results/run_2_predicted.csv')

# # remove all rows from data2 where 'audio_prob' is empty
# data2 = data2.dropna(subset=['audio_prob'])
# data2['checkpoint'] = data2['checkpoint'].astype(int)

# # Concatenate data1 and data2 vertically
# results = pd.concat([data1, data2], axis=0)

# # save complete results to csv
# results.to_csv('multimodal_results/run_2_complete_results.csv', index=False)
## ============================================================================

## ======================= TESTING COMBINED RUN 15 ============================

results = pd.read_csv('multimodal_results/combined_run_15.csv')

# Drop rows with Nan values in audio_prob_class or video_prob
results = results.dropna(subset=['audio_prob_class', 'video_prob'])

# Check shape
print("Results shape")
print(results.shape)
print()

audio_string_probs = results['audio_prob_class']
video_string_probs = results['video_prob']

# check data types

video_probs = convert_strings_to_arrays(video_string_probs)

video_probs_normalized=[]

# Drop the 4th index in 'video_probs'
for i in range(len(video_probs)):
    video_probs_normalized.append(np.delete(video_probs[i], 4))

video_probs_normalized = [(video_probs_normalized[i]/np.sum(video_probs_normalized[i])).reshape(1,6) for i in range(len(video_probs_normalized))]

audio_probs = convert_strings_to_arrays(audio_string_probs, pad_to_length=6)

results['audio_prob'] = audio_probs
results['video_prob'] = video_probs_normalized

# Use binary mask to change every instance of '5' to '4' in 'Label'
results['Label'] = results['Label'].replace(5, 4)
results['Label'] = results['Label'].replace(6, 5)

## ============================================================================
## ============================================================================
## ============================================================================
## ========== SPLITTING TRAIN AND TEST DATA ===================================

# num_files_per_class = 120
# grouped = results.groupby('Emotion')

# # For each group, check if the group size is less than num_files_per_class
# for emotion, group in grouped:
#     if len(group) < num_files_per_class:
#         print(f"Warning: There are only {len(group)} files for emotion {emotion}, less than the desired {num_files_per_class}.")
#         num_files_per_class = len(group)

# # Randomly select num_files_per_class rows from each group
# selected_df = grouped.apply(lambda x: x.sample(min(len(x), num_files_per_class)))

# # Reset the index of the selected DataFrame
# selected_df.reset_index(drop=True, inplace=True)

# # Sort the DataFrame by the first column
# selected_df.sort_values('filename', inplace=True)

# # Get the list of selected files
# selected_files = selected_df['filename'].tolist()

# Get selected_files list from pickle file
with open('multimodal_results/multimodal_baseline_test_set.pkl', 'rb') as f:
    selected_files = pickle.load(f)

test_data = results[results['filename'].isin(selected_files)]

train_data = results[~results['filename'].isin(selected_files)]

# Check shapes
print("Train data shape:")
print(train_data.shape)
print()
print("Test data shape:")
print(test_data.shape)
print()

## ============================================================================
## ============================================================================
## ============================================================================

# Drop columns checkpoint audio_prob_class audio_prob_reg audio_prob_mult
train_data = train_data.drop(columns=['checkpoint', 'audio_prob_class', 'audio_prob_reg', 'audio_prob_mult'])
# print(train_data.head(20))
# print()

audio_probs = train_data['audio_prob'].tolist()
video_probs = train_data['video_prob'].tolist()

all_audio_probs = np.concatenate(audio_probs, axis=0) # Shape: (n_samples, 7)
all_video_probs = np.concatenate(video_probs, axis=0) # Shape: (n_samples, 7)

# One hot encoding of the true labels
# Define your list of emotion labels (as an example)
emotion_labels = train_data['Emotion']
# print("Unique values in emotion_labels:")
# print(emotion_labels.unique())
# my_encoding_dict_model = {'neu': 0, 'ang': 1, 'hap': 2, 'sad': 3, 'sur': 4, 'fea': 5, 'dis': 6}
my_encoding_dict_model = {'neu': 0, 'ang': 1, 'hap': 2, 'sad': 3, 'fea': 4, 'dis': 5} # removed surprise
num_categories = len(my_encoding_dict_model)
one_hot_encoded = np.zeros((len(emotion_labels), num_categories))

# Fill the appropriate elements with 1
for idx, label in enumerate(emotion_labels):
    one_hot_encoded[idx, my_encoding_dict_model[label]] = 1

# print(one_hot_encoded)

# Check that all shapes are correct
print("-------------------------------------------------")
print("Check shapes")
print("-------------------------------------------------")
print(f"Audio probs: {all_audio_probs.shape}")
print(f"Video probs: {all_video_probs.shape}")
print(f"True labels: {one_hot_encoded.shape}")



# Use binary mask to get index of highest probability for audio_prob in each row
audio_predictions = np.argmax(all_audio_probs, axis=1)

# Inspect audio_predictions (shape, type, first element)
# print("Inspecting audio predictions (shape, type, first element)")
# print(audio_predictions.shape)
# print(type(audio_predictions))
# print(audio_predictions[0])
# print()

train_data['audio_predicted_single'] = audio_predictions

# Similarily create a column for video predictions
video_predictions = np.argmax(all_video_probs, axis=1)

# Inspect video_predictions (shape, type, first element)
# print("Inspecting video predictions (shape, type, first element)")
# print(video_predictions.shape)
# print(type(video_predictions))
# print(video_predictions[0])
# print()

train_data['video_predicted_single'] = video_predictions

# Print the first 10 rows of the columns "audio_predicted_single" and "video_predicted_single"
print(train_data[['Label', 'audio_predicted_single', 'video_predicted_single']].head(10))

# print("-------------------------------------------------")
# print("One weight for each emotion - grid search")
# print("-------------------------------------------------")
# # Now use separate weights for each emotion
# # Define a range of possible weight values (for simplicity, consider only a few steps)
# weight_values = np.arange(0, 1.1, 0.1)  # 5 steps from 0 to 1

# # Initialize the best score and corresponding weights
# best_accuracy = 0
# best_weights = (0, 0)

# # Loop over all possible combinations of weights for audio and video
# num_times_ran = 0
# for w_a0 in weight_values:
#     for w_a1 in weight_values:
#         for w_a2 in weight_values:
#             for w_a3 in weight_values:
#                 # for w_a4 in weight_values:
#                 # for w_a5 in weight_values:
#                 # for w_a6 in weight_values:
#                 w_a = np.array([w_a0, w_a1, w_a2, w_a3, 0, 0, 0])
#                 w_v = 1 - w_a
#                 accuracy = evaluate(w_a, w_v, all_audio_probs, all_video_probs, one_hot_encoded)
#                 if accuracy > best_accuracy:
#                     best_accuracy = accuracy
#                     best_weights = (w_a, w_v)
#                 num_times_ran += 1

# print("Best weights:")
# print(f"w_a={best_weights[0]}")
# print(f"w_v={best_weights[1]}")
# print("Accuracy:", best_accuracy)
# print(f"Number of times ran: {num_times_ran}")
# print("-------------------------------------------------")


# print("-------------------------------------------------")
# print("One weight for each emotion - optimization MSE")
# print("-------------------------------------------------")
# # Now use a more sophisticated optimization algorithm
# # Initial weights for the first four categories (8 weights total)
# initial_weights = np.ones(4) * 0.5

# # Bounds for these weights
# bounds = [(0, 1)] * 4

# # Optimize
# result = minimize(loss_function, initial_weights, args=(all_audio_probs, all_video_probs, one_hot_encoded), bounds=bounds, method='L-BFGS-B')

# weights_audio = np.concatenate([result.x, [0, 0, 0]])
# weights_video = 1 - weights_audio

# print("Optimized weights for audio:", weights_audio)
# print("Optimized weights for video:", weights_video)
# print("Minimum loss:", result.fun)

# best_w_a = weights_audio
# best_w_v = weights_video
# print("best_w_a_shape: ", best_w_a.shape)
# # Type
# print(type(best_w_a))
# print("best_w_a: ", best_w_a)
# print("best_w_v_shape: ", best_w_v.shape)
# # Type
# print(type(best_w_v))
# print("best_w_v: ", best_w_v)
# best_w_a = best_w_a.reshape(1,7)
# best_w_v = best_w_v.reshape(1,7)

# # Save best weights to pickle file

# # with open('best_weights.pkl', 'wb') as f:
#     # pickle.dump([best_w_a, best_w_v], f)

# # Load weights from pickle file
# # with open('best_weights.pkl', 'rb') as f:
#     # best_w_a, best_w_v = pickle.load(f)

# best_accuracy = evaluate(best_w_a, best_w_v, all_audio_probs, all_video_probs, one_hot_encoded)
# print("Best accuracy: ", best_accuracy)
# print("-------------------------------------------------")

print("-------------------------------------------------")
print("One weight for each emotion - optimization Cross ENTROPY")
print("-------------------------------------------------")
# Now use a more sophisticated optimization algorithm
# Initial weights for the first four categories (8 weights total)
initial_weights = np.ones(4) * 0.5

# Bounds for these weights
bounds = [(0, 1)] * 4

# Optimize
result = minimize(loss_function_CE, initial_weights, args=(all_audio_probs, all_video_probs, one_hot_encoded), bounds=bounds, method='L-BFGS-B')

weights_audio = np.concatenate([result.x, [0, 0]])
weights_video = 1 - weights_audio

print("Optimized weights for audio:", weights_audio)
print("Optimized weights for video:", weights_video)
print("Minimum loss:", result.fun)

wa = weights_audio
wv = weights_video
print("best_w_a_shape: ", wa.shape)
# Type
print(type(wa))
print("best_w_a: ", wa)
print("best_w_v_shape: ", wv.shape)
# Type
print(type(wv))
print("best_w_v: ", wv)
wa = wa.reshape(1,6)
wv = wv.reshape(1,6)

# Save best weights to pickle file

with open('multimodal_results/best_weights_baseline.pkl', 'wb') as f:
    pickle.dump([wa, wv], f)

# # Load weights from pickle file
# # with open('best_weights.pkl', 'rb') as f:
#     # best_w_a, best_w_v = pickle.load(f)

best_accuracy = evaluate(wa, wv, all_audio_probs, all_video_probs, one_hot_encoded)
print("Best accuracy: ", best_accuracy)

# Apply the weights to all probabilities
# wa * all_audio_probs + wv * all_video_probs
combined_probs = wa * all_audio_probs + wv * all_video_probs

# Check combined_probs (shape, type, first element)
print("Inspecting combined probs (shape, type, first element)")
print(combined_probs.shape)
print(type(combined_probs))
print(combined_probs[0])
print()

combined_predictions = np.argmax(combined_probs, axis=1)

# Add to train_data dataframe
train_data['combined_predicted'] = combined_predictions

# Print columns of train_data
print("Columns of train_data:")
print(train_data.columns)

audio_probs = test_data['audio_prob'].tolist()
video_probs = test_data['video_prob'].tolist()

all_audio_probs = np.concatenate(audio_probs, axis=0)
all_video_probs = np.concatenate(video_probs, axis=0)

audio_predictions = np.argmax(all_audio_probs, axis=1)
test_data.loc[:, 'audio_predicted_single'] = audio_predictions

video_predictions = np.argmax(all_video_probs, axis=1)
test_data.loc[:, 'video_predicted_single'] = video_predictions

combined_probs = wa * all_audio_probs + wv * all_video_probs
combined_predictions = np.argmax(combined_probs, axis=1)
test_data.loc[:, 'combined_predicted'] = combined_predictions

# create confusion matrix
create_confusion_matrix(test_data)

print("-------------------------------------------------")
print("Single modality accuracies")
print("-------------------------------------------------")

emotion_labels = test_data['Emotion']
# print("Unique values in emotion_labels:")
# print(emotion_labels.unique())
# my_encoding_dict_model = {'neu': 0, 'ang': 1, 'hap': 2, 'sad': 3, 'sur': 4, 'fea': 5, 'dis': 6}
my_encoding_dict_model = {'neu': 0, 'ang': 1, 'hap': 2, 'sad': 3, 'fea': 4, 'dis': 5} # removed surprise
num_categories = len(my_encoding_dict_model)
one_hot_encoded = np.zeros((len(emotion_labels), num_categories))

# Fill the appropriate elements with 1
for idx, label in enumerate(emotion_labels):
    one_hot_encoded[idx, my_encoding_dict_model[label]] = 1

only_video_accuracy = get_single_modality_accuracy(all_video_probs, one_hot_encoded)
only_audio_accuracy = get_single_modality_accuracy(all_audio_probs, one_hot_encoded)
print(f"Only video accuracy: {only_video_accuracy}")
print(f"Only audio accuracy: {only_audio_accuracy}")
print()

# print("-------------------------------------------------")



# print("-------------------------------------------------")
# print("One weight for each emotion - optimization ACCURACY")
# print("-------------------------------------------------")
# # Now use a more sophisticated optimization algorithm
# # Initial weights for the first four categories (8 weights total)
# initial_weights = np.ones(4) * 0

# # Bounds for these weights
# bounds = [(0, 1)] * 4

# # Optimize
# result = minimize(optimize_accuracy, initial_weights, args=(all_audio_probs, all_video_probs, one_hot_encoded), bounds=bounds, method='Powell', 
#                   options={'maxiter': 2000})

# weights_audio = np.concatenate([result.x, [0, 0, 0]])
# weights_video = 1 - weights_audio

# print("Optimized weights for audio:", weights_audio)
# print("Optimized weights for video:", weights_video)
# print("Optimal accuracy:", result.fun)

# best_w_a = weights_audio
# best_w_v = weights_video
# print("best_w_a_shape: ", best_w_a.shape)
# # Type
# print(type(best_w_a))
# print("best_w_a: ", best_w_a)
# print("best_w_v_shape: ", best_w_v.shape)
# # Type
# print(type(best_w_v))
# print("best_w_v: ", best_w_v)
# best_w_a = best_w_a.reshape(1,7)
# best_w_v = best_w_v.reshape(1,7)

# # Save best weights to pickle file

# # with open('best_weights.pkl', 'wb') as f:
#     # pickle.dump([best_w_a, best_w_v], f)

# # Load weights from pickle file
# # with open('best_weights.pkl', 'rb') as f:
#     # best_w_a, best_w_v = pickle.load(f)

# best_accuracy = evaluate(best_w_a, best_w_v, all_audio_probs, all_video_probs, one_hot_encoded)
# print("Best accuracy: ", best_accuracy)
# print("-------------------------------------------------")