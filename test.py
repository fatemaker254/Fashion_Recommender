from cv2 import norm
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50, preprocess_input
import tensorflow
from keras.layers import GlobalMaxPooling2D
from PIL import Image
import tqdm as tq
import matplotlib.pyplot as plt

# Load precomputed embeddings and filenames
embeddings = pickle.load(open("embeddings.pkl", "rb"))
filenames = pickle.load(open("filenames.pkl", "rb"))
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])


def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result


def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))


similarities = []


# Function to recommend images based on input image
def recommend_images(input_img_path, top_n=5):
    input_embedding = extract_features(input_img_path, model)
    # similarities = []

    for emb in tq.tqdm(embeddings):
        sim = cosine_similarity(input_embedding, emb)
        similarities.append(sim)

    sorted_indices = np.argsort(similarities)[::-1][:top_n]
    recommended_images = [(filenames[i], similarities[i]) for i in sorted_indices]

    return recommended_images


# Path to the input image
input_image_path = "images2/2003.jpg"

# Number of top recommended images
top_n = 5

recommended_images = recommend_images(input_image_path, top_n)
for i, (image_path, similarity) in enumerate(recommended_images):
    print(f"Recommended image {i+1}: {image_path} (Similarity: {similarity*100:.2f}%)")

# Plotting the accuracy curve
plt.plot(np.arange(len(similarities)), similarities)
plt.xlabel("Image Index")
plt.ylabel("Similarity Score")
plt.title("Accuracy Curve for Image Recommendations")
plt.show()
