import dataset
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

dataset = dataset.Dataset(data=[{"text": "Example sentence 1."}, {"text": "Example sentence 2."}])

userQuery = input("Enter your query: ")

queryVec = model.encode(userQuery,convert_to_tensor=True)

# Loop over each document and calculate similarity
for idx, item in enumerate(dataset.get_data()):
    dotProd = np.dot(queryVec, list(dataset.get_embeddings())[idx])
    normVec1 = norm(queryVec)
    normVec2 = norm(list(dataset.get_embeddings())[idx])
    
    similarity = dotProd / (normVec1 * normVec2)
    
    print(f"Text: {item['text']}")
    print(f"Similarity: {similarity}")
    print()