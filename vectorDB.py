from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm

from dataset import Dataset

class vectorDB(Dataset):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    def __init__(self, data):
        self.data = data
        texts = [item['text'] for item in data]
        self.datasetVecs = self.model.encode(texts)
        self.__randomHashVecArray = np.random.rand(128, 384)
        self.datasetHashArray = np.dot(self.datasetVecs, self.__randomHashVecArray.T)
        self.binaryDatasetHashArray = (self.datasetHashArray > 0).astype(int)
    
    def search(self, query, k=5):
        """Approximate search using Hamming distance over binary hashes.

        Returns a dict containing the best match (for backwards
        compatibility) and the top-k matches sorted by increasing
        Hamming distance.
        """

        self.queryVec = self.model.encode(query)
        self.binaryQueryHash = (np.dot(self.queryVec, self.__randomHashVecArray.T) > 0).astype(int)

        # Vectorized Hamming distance over all dataset hashes
        # (XOR then count differing bits per row)
        distances = np.sum(self.binaryDatasetHashArray != self.binaryQueryHash, axis=1)

        # Get indices of top-k closest hashes
        k = min(k, len(self.data))
        top_indices = np.argsort(distances)[:k]

        matches = []
        for idx in top_indices:
            matches.append({
                "text": self.data[idx]['text'],
                "index": int(idx),
                "hamming_distance": int(distances[idx])
            })

        if matches:
            best = matches[0]
            return {
                "best_match": best["text"],
                "hamming_distance": best["hamming_distance"],
                "matches": matches
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