from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm

class vectorDB:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    def __init__(self, data):
        self.data = data
        texts = [item['text'] for item in data]
        self.datasetVecs = self.model.encode(texts)
        self.__randomHashVecArray = np.random.rand(32, 384)
        self.datasetHashArray = np.dot(self.datasetVecs, self.__randomHashVecArray.T)
        self.binaryDatasetHashArray = (self.datasetHashArray > 0).astype(int)
    
    def search(self, query):
        self.queryVec = self.model.encode(query)
        self.binaryQueryHash = (np.dot(self.queryVec, self.__randomHashVecArray.T) > 0).astype(int)
        # compare binary hashes by finding the one with the minimum Hamming distance
        min_hamming_dist = float('inf')
        best_match_idx = -1

        for idx, binary_hash in enumerate(self.binaryDatasetHashArray):
            # Calculate Hamming distance
            hamming_dist = np.sum(self.binaryQueryHash != binary_hash)
            if hamming_dist < min_hamming_dist:
                min_hamming_dist = hamming_dist
                best_match_idx = idx

        if best_match_idx != -1:
            return {
                "best_match": self.data[best_match_idx]['text'],
                "hamming_distance": min_hamming_dist
            }
        else:
            return None
    
    def brute_force_search(self, query):
        self.queryVec = self.model.encode(query)
        normVec1 = norm(self.queryVec)
        results = []

        for idx, item in enumerate(self.data):
            dotProd = np.dot(self.queryVec, self.datasetVecs[idx])
            normVec2 = norm(self.datasetVecs[idx])
            
            similarity = dotProd / (normVec1 * normVec2)
            
            results.append({
                "text": item['text'],
                "similarity": similarity
            })
        
        return results