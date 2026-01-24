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
    {"text": "Apples are a popular and healthy snack."}
]


dataset = dataset.Dataset(data=data)

userQuery = input("Enter your query: ")

queryVec = model.encode(userQuery)
datasetVecs = list(dataset.get_embeddings())

normVec1 = norm(queryVec)


# Loop over each document and calculate similarity
for idx, item in enumerate(dataset.get_data()):
    dotProd = np.dot(queryVec, datasetVecs[idx])
    normVec2 = norm(datasetVecs[idx])
    
    similarity = dotProd / (normVec1 * normVec2)
    
    print(f"Text: {item['text']}")
    print(f"Similarity: {similarity}")
    print()