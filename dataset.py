from sentence_transformers import SentenceTransformer

class Dataset:
    def __init__(self, data, model_name='all-MiniLM-L6-v2'):
        self.data = data
        self.model = SentenceTransformer(model_name)
        self.embeddings = self._compute_embeddings()

    def _compute_embeddings(self):
        texts = [item['text'] for item in self.data]
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings

    def get_data(self):
        return self.data

    def get_embeddings(self):
        return self.embeddings