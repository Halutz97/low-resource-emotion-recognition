# This script reads and processes the results of the multimodal experiments
# Furthermore, a logic for determining the best model is implemented
# That is, automatically adjusting the audio and visual weights

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def convert_strings_to_arrays(list_of_strings):
    """
    Convert probabiliteis given as strings to numpy arrays (1,7)
    :param list_of_strings: list of strings
    :return: list of numpy arrays
    """
    list_of_arrays = []
    for probabilities in list_of_strings:
        probabilities = probabilities.split(" ")
        probabilities = [x.replace('[', '') for x in probabilities]
        probabilities = [x.replace(']', '') for x in probabilities]
        probabilities = [x.replace('\n', '') for x in probabilities]
        probabilities = [x for x in probabilities if x] # remove empty strings
        probs_float = [float(x) for x in probabilities]
        probs_float = np.array(probs_float).reshape(1,7)
        list_of_arrays.append(probs_float)
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

def loss_function(weights, *args):
    # Unpack arguments
    audio_probs, video_probs, true_labels = args

    # Assuming we are optimizing the first four weights for each modality
    w_a = np.concatenate([weights[:4], [0, 0, 0]])  # Last three weights for audio are zero
    w_v = np.concatenate([weights[4:], [1, 1, 1]])  # Last three weights for video are one

    # Calculate the combined probabilities
    combined_probs = w_a * audio_probs + w_v * video_probs

    # Example: Mean Squared Error
    mse = np.mean((combined_probs - true_labels) ** 2)
    return mse

def get_single_modality_accuracy(predictions, true_labels):
    predictions = np.argmax(predictions, axis=1)
    true_classes = np.argmax(true_labels, axis=1)
    accuracy = np.mean(predictions == true_classes)
    return accuracy

# Read the results
results = pd.read_csv('multimodal_results/run_2_predicted_checkpoint_900.csv')

print(results.head())

audio_string_probs = results['audio_prob']
video_string_probs = results['video_prob']
audio_probs = convert_strings_to_arrays(audio_string_probs)
video_probs = convert_strings_to_arrays(video_string_probs)

results['audio_prob'] = audio_probs
results['video_prob'] = video_probs

print()
print("Inspecting audio probs (shape, type, first element)")
print(audio_probs[0].shape)
print(type(audio_probs[0]))
print(audio_probs[0])
print()

all_audio_probs = np.concatenate(audio_probs, axis=0) # Shape: (n_samples, 7)
all_video_probs = np.concatenate(video_probs, axis=0) # Shape: (n_samples, 7)

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
my_encoding_dict_model = {'neu': 0, 'ang': 1, 'hap': 2, 'sad': 3, 'sur': 4, 'fea': 5, 'dis': 6}
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

print("-------------------------------------------------")
print("One weight per modality - grid search")
print("-------------------------------------------------")

best_accuracy = 0
best_weights = (0, 0)

# Grid search
num_times_ran = 0
for w_v in np.arange(0, 1.001, 0.001):
    w_a = 1 - w_v
    accuracy = evaluate(w_a, w_v, all_audio_probs, all_video_probs, one_hot_encoded)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_weights = (w_a, w_v)
    num_times_ran += 1

print(f"Number of times ran: {num_times_ran}")
print(f"Best Weights:")
print(f"w_a={best_weights[0]}")
print(f"w_v={best_weights[1]}")
print(f"Accuracy: {best_accuracy}")
print("-------------------------------------------------")


print("-------------------------------------------------")
print("One weight for each emotion - grid search")
print("-------------------------------------------------")
# Now use separate weights for each emotion
# Define a range of possible weight values (for simplicity, consider only a few steps)
weight_values = np.arange(0, 1.1, 0.1)  # 5 steps from 0 to 1

# Initialize the best score and corresponding weights
best_accuracy = 0
best_weights = (0, 0)

# Loop over all possible combinations of weights for audio and video
num_times_ran = 0
for w_a0 in weight_values:
    for w_a1 in weight_values:
        for w_a2 in weight_values:
            for w_a3 in weight_values:
                # for w_a4 in weight_values:
                # for w_a5 in weight_values:
                # for w_a6 in weight_values:
                w_a = np.array([w_a0, w_a1, w_a2, w_a3, 0, 0, 0])
                w_v = 1 - w_a
                accuracy = evaluate(w_a, w_v, all_audio_probs, all_video_probs, one_hot_encoded)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_weights = (w_a, w_v)
                num_times_ran += 1

print("Best weights:")
print(f"w_a={best_weights[0]}")
print(f"w_v={best_weights[1]}")
print("Accuracy:", best_accuracy)
print(f"Number of times ran: {num_times_ran}")
print("-------------------------------------------------")


print("-------------------------------------------------")
print("One weight for each emotion - optimization")
print("-------------------------------------------------")
# Now use a more sophisticated optimization algorithm
# Initial weights for the first four categories (8 weights total)
initial_weights = np.ones(8) * 0.5

# Bounds for these weights
bounds = [(0, 1)] * 8

# Optimize
result = minimize(loss_function, initial_weights, args=(all_audio_probs, all_video_probs, one_hot_encoded), bounds=bounds)

print("Optimized weights for audio:", np.concatenate([result.x[:4], [0, 0, 0]]))
print("Optimized weights for video:", np.concatenate([result.x[4:], [1, 1, 1]]))
print("Minimum loss:", result.fun)

best_w_a = np.concatenate([result.x[:4], [0, 0, 0]])
best_w_v = np.concatenate([result.x[4:], [1, 1, 1]])
print("best_w_a_shape: ", best_w_a.shape)
# Type
print(type(best_w_a))
print("best_w_a: ", best_w_a)
print("best_w_v_shape: ", best_w_v.shape)
# Type
print(type(best_w_v))
print("best_w_v: ", best_w_v)
best_w_a = best_w_a.reshape(1,7)
best_w_v = best_w_v.reshape(1,7)

best_accuracy = evaluate(best_w_a, best_w_v, all_audio_probs, all_video_probs, one_hot_encoded)
print("Best accuracy: ", best_accuracy)
print("-------------------------------------------------")