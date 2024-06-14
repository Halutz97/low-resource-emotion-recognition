# This script reads and processes the results of the multimodal experiments
# Furthermore, a logic for determining the best model is implemented
# That is, automatically adjusting the audio and visual weights

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import pickle
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt


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
    # plt.savefig('Ari_figures/confusion_matrix_FER_validation_voted.png', dpi=300, bbox_inches='tight')
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
    w_a = np.concatenate([weights, [0, 0, 0]])  # Last three weights for audio are zero
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

print(results.head())
print()
# Print column names of results
print("Column names:")
print(results.columns)
print()

# Print rows with Nan values in audio_prob_class or video_prob
print(results[results['audio_prob_class'].isnull()])
print(results[results['video_prob'].isnull()])

# Drop rows with Nan values in audio_prob_class or video_prob
results = results.dropna(subset=['audio_prob_class', 'video_prob'])

# Check shape
print("New shape")
print(results.shape)
print()

audio_string_probs = results['audio_prob_class']
video_string_probs = results['video_prob']

# check data types
print("Data types:")
print(type(audio_string_probs[0]))
print(type(video_string_probs[0]))
print(audio_string_probs[0])
print()

video_probs = convert_strings_to_arrays(video_string_probs)

# print(video_probs[0].shape)
# sum
# print(np.sum(video_probs[0]))

video_probs_normalized=[]

# Drop the 4th index in 'video_probs'
for i in range(len(video_probs)):
    video_probs_normalized.append(np.delete(video_probs[i], 4))

# Check the shape of the first element in video_probs_normalized
# print(video_probs_normalized[0].shape)    
# Print sum of the elements of the first element in video_probs_normalized
# print(np.sum(video_probs_normalized[0]))
# Normalize video_probs_normalized
video_probs_normalized = [(video_probs_normalized[i]/np.sum(video_probs_normalized[i])).reshape(1,6) for i in range(len(video_probs_normalized))]
# reshape
# video_probs_normalized = [video_probs_normalized[i].reshape(1,6) for i in range(len(video_probs_normalized))]
# Check the shape of the first element in video_probs_normalized
# print(video_probs_normalized[0].shape)
# Print sum of the elements of the first element in video_probs_normalized
# print(np.sum(video_probs_normalized[0]))
# print(video_probs[0])
# print(video_probs_normalized[0])

# Type
# print(type(video_probs_normalized[0]))

audio_probs = convert_strings_to_arrays(audio_string_probs, pad_to_length=6)

results['audio_prob'] = audio_probs
results['video_prob'] = video_probs_normalized

# Drop columns checkpoint audio_prob_class audio_prob_reg audio_prob_mult
results = results.drop(columns=['checkpoint', 'audio_prob_class', 'audio_prob_reg', 'audio_prob_mult'])
print(results.head(20))
print()

print()
print("Inspecting audio probs (shape, type, first element)")
print(audio_probs[0].shape)
print(type(audio_probs[0]))
print(audio_probs[0])
print()

print("Inspecting video probs (shape, type, first element)")
print(video_probs_normalized[0].shape)
print(type(video_probs_normalized[0]))
print(video_probs_normalized[0])
print()

all_audio_probs = np.concatenate(audio_probs, axis=0) # Shape: (n_samples, 7)
all_video_probs = np.concatenate(video_probs_normalized, axis=0) # Shape: (n_samples, 7)

print()
print(all_audio_probs.shape)
print(all_video_probs.shape)
print(all_audio_probs[0].shape)
print(all_audio_probs[0])
print(audio_probs[0])
print()

# One hot encoding of the true labels
# Define your list of emotion labels (as an example)
emotion_labels = results['Emotion']
print("Unique values in emotion_labels:")
print(emotion_labels.unique())
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

# checker = np.argmax(one_hot_encoded, axis=1)
# print(checker)
# print(checker.shape)

print("-------------------------------------------------")
print("Single modality accuracies")
print("-------------------------------------------------")

only_video_accuracy = get_single_modality_accuracy(all_video_probs, one_hot_encoded)
only_audio_accuracy = get_single_modality_accuracy(all_audio_probs, one_hot_encoded)
print(f"Only video accuracy: {only_video_accuracy}")
print(f"Only audio accuracy: {only_audio_accuracy}")
print()

# Use binary mask to get index of highest probability for audio_prob in each row
audio_predictions = np.argmax(all_audio_probs, axis=1)

# Inspect audio_predictions (shape, type, first element)
# print("Inspecting audio predictions (shape, type, first element)")
# print(audio_predictions.shape)
# print(type(audio_predictions))
# print(audio_predictions[0])
# print()

results['audio_predicted_single'] = audio_predictions

# Similarily create a column for video predictions
video_predictions = np.argmax(all_video_probs, axis=1)

# Inspect video_predictions (shape, type, first element)
# print("Inspecting video predictions (shape, type, first element)")
# print(video_predictions.shape)
# print(type(video_predictions))
# print(video_predictions[0])
# print()

results['video_predicted_single'] = video_predictions

# Print the first 10 rows of the columns "audio_predicted_single" and "video_predicted_single"
print(results[['Label', 'audio_predicted_single', 'video_predicted_single']].head(10))

# print("-------------------------------------------------")
# print("One weight per modality - grid search")
# print("-------------------------------------------------")

# best_accuracy = 0
# best_weights = (0, 0)

# # Grid search
# num_times_ran = 0
# for w_a in np.arange(0, 1.0005, 0.0005):
#     w_v = 1 - w_a
#     accuracy = evaluate(w_a, w_v, all_audio_probs, all_video_probs, one_hot_encoded)
#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         best_weights = (w_a, w_v)
#     num_times_ran += 1

# print(f"Number of times ran: {num_times_ran}")
# print(f"Best Weights:")
# print(f"w_a={best_weights[0]}")
# print(f"w_v={best_weights[1]}")
# print(f"Accuracy: {best_accuracy}")
# print("-------------------------------------------------")


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

# print("-------------------------------------------------")
# print("One weight for each emotion - optimization Cross ENTROPY")
# print("-------------------------------------------------")
# # Now use a more sophisticated optimization algorithm
# # Initial weights for the first four categories (8 weights total)
# initial_weights = np.ones(4) * 0.5

# # Bounds for these weights
# bounds = [(0, 1)] * 4

# # Optimize
# result = minimize(loss_function_CE, initial_weights, args=(all_audio_probs, all_video_probs, one_hot_encoded), bounds=bounds, method='L-BFGS-B')

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

# with open('best_weights.pkl', 'wb') as f:
#     pickle.dump([best_w_a, best_w_v], f)

# # Load weights from pickle file
# # with open('best_weights.pkl', 'rb') as f:
#     # best_w_a, best_w_v = pickle.load(f)

# best_accuracy = evaluate(best_w_a, best_w_v, all_audio_probs, all_video_probs, one_hot_encoded)
# print("Best accuracy: ", best_accuracy)
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