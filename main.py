import dataset
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
data = [
    {"text": "The sky is blue and vast."},
    {"text": "An orange is a type of citrus fruit."},
    {"text": "Formula 1 is the pinnacle of motorsport."},
    {"text": "The ocean covers most of the Earth's surface."},
    {"text": "Apples are a popular and healthy snack."},
    {"text": "Mountains are majestic landforms that rise prominently above their surroundings."},
    {"text": "Soccer is the most popular sport worldwide."},
    {"text": "Bananas are rich in potassium and fiber."},
    {"text": "The desert is characterized by its arid conditions and sparse vegetation."},
    {"text": "Tennis is played on various surfaces including clay, grass, and hard courts."},
    {"text": "Grapes can be eaten fresh or used to make wine."},
    {"text": "Rivers are natural flowing watercourses that usually lead to an ocean, sea, or lake."},
    {"text": "Basketball is a fast-paced game played with a hoop and a ball."},
    {"text": "Cherries are small, round fruits that can be sweet or tart."},
    {"text": "Forests are dense collections of trees and undergrowth."},
    {"text": "Baseball is known as America's pastime."},
    {"text": "Pineapples have a unique sweet and tangy flavor."},
    {"text": "Volcanoes are openings in the Earth's crust that allow molten rock to escape."},
    {"text": "Hockey is played on ice with skates and a puck."},
    {"text": "Strawberries are bright red fruits that are often used in desserts."},
    {"text": "Canyons are deep gorges typically carved by rivers over time."}
]


dataset = dataset.Dataset(data=data)

userQuery = input("Enter your query: ")

queryVec = model.encode(userQuery)
datasetVecs = list(dataset.get_embeddings())

normVec1 = norm(queryVec)

randomHashVecArray = np.random.rand(32, 384)

QueryHash = np.dot(queryVec, randomHashVecArray.T)
print(f"Query Hash: {QueryHash}")

bianaryQueryHash = (QueryHash > 0).astype(int)
print(f"Binary Query Hash: {bianaryQueryHash}")

datasetHashArray = np.dot(datasetVecs, randomHashVecArray.T)
 
binaryDatasetHashArray = (datasetHashArray > 0).astype(int)
print(f"Binary Dataset Hash Array: {binaryDatasetHashArray}")


# compare binary hashes by finding the one with the minimum Hamming distance
min_hamming_dist = float('inf')
best_match_idx = -1

for idx, binary_hash in enumerate(binaryDatasetHashArray):
    # Calculate Hamming distance
    hamming_dist = np.sum(bianaryQueryHash != binary_hash)
    if hamming_dist < min_hamming_dist:
        min_hamming_dist = hamming_dist
        best_match_idx = idx

if best_match_idx != -1:
    print(f"\nBest LSH match found with document: {dataset.get_data()[best_match_idx]['text']}")
    print(f"Hamming distance: {min_hamming_dist}")


# Loop over each document and calculate similarity (brute force)
for idx, item in enumerate(dataset.get_data()):
    dotProd = np.dot(queryVec, datasetVecs[idx])
    normVec2 = norm(datasetVecs[idx])
    
    similarity = dotProd / (normVec1 * normVec2)
    
    print(f"Text: {item['text']}")
    print(f"Similarity: {similarity}")
    print()