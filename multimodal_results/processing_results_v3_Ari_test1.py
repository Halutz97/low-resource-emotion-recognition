# This script reads and processes the results of the multimodal experiments
# Furthermore, a logic for determining the best model is implemented
# That is, automatically adjusting the audio and visual weights

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import pickle
from scipy.optimize import basinhopping

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
    # print()
    # print("-------------------------------------------------")
    # print("----- INSIDE LOSS FUNCTION CROSS ENTROPY --------")
    # print("-------------------------------------------------")
    # print()
    
    # Unpack arguments
    audio_probs, video_probs, true_labels = args

    # print shapes of all
    # print("audio_probs shape: ", audio_probs.shape)
    # print("video_probs shape: ", video_probs.shape)
    # print("true_labels shape: ", true_labels.shape)
    # print()

    # Print first element of each, rounded to 2 decimal places
    # print("audio_probs first element: ", np.round(audio_probs[0], 2))
    # print("video_probs first element: ", np.round(video_probs[0], 2))
    # print("true_labels first element: ", np.round(true_labels[0], 2))
    # print()

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

    # print()
    # print("-------------------------------------------------")
    # print("----- CROSS ENTROPY LOSS EXITING-----------------")
    # print("-------------------------------------------------")
    # print()

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

results = pd.read_csv('multimodal_results/run_2_complete_results.csv')

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

print("-----------------------------------------------------------")
print("One weight for each emotion - optimization CROSS ENTROPY")
print("-----------------------------------------------------------")
# Now use a more sophisticated optimization algorithm
# Initial weights for the first four categories (8 weights total)
# initial_weights = np.ones(4) * 1
initial_weights = np.array([0.4, 0.6, 0.0, 0.3])

print("Initial weights: ", str(initial_weights))
 
initial_evaluate_weights = np.concatenate([initial_weights, [0, 0, 0]])
initial_accuracy = evaluate(initial_evaluate_weights, 1 - initial_evaluate_weights, all_audio_probs, all_video_probs, one_hot_encoded)
print("Initial accuracy: ", np.round(initial_accuracy,3))

# Bounds for these weights
bounds = [(0, 1)] * 4

minimizer_kwargs = {
    "method": "Powell",
    "bounds": bounds,
    "args": (all_audio_probs, all_video_probs, one_hot_encoded),
}

# Call basinhopping
result = basinhopping(loss_function_CE, initial_weights, minimizer_kwargs=minimizer_kwargs, niter=1)

# Optimize
# result = minimize(loss_function_CE, initial_weights, args=(all_audio_probs, all_video_probs, one_hot_encoded), bounds=bounds, method='Basinhopping')

weights_audio = np.concatenate([result.x, [0, 0, 0]])
weights_video = 1 - weights_audio

# print("Optimized weights for audio:", weights_audio)
# print("Optimized weights for video:", weights_video)
# print("Minimum loss:", result.fun)

best_w_a = weights_audio
best_w_v = weights_video
# round weights to 3 decimal places
best_w_a = np.round(best_w_a, 3)
best_w_v = np.round(best_w_v, 3)
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

# Save best weights to pickle file

# with open('best_weights.pkl', 'wb') as f:
    # pickle.dump([best_w_a, best_w_v], f)

# Load weights from pickle file
# with open('best_weights.pkl', 'rb') as f:
    # best_w_a, best_w_v = pickle.load(f)

best_accuracy = evaluate(best_w_a, best_w_v, all_audio_probs, all_video_probs, one_hot_encoded)
print("Best accuracy: ", np.round(best_accuracy,3))
print("-------------------------------------------------")