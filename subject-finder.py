import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from subjects import *

class SubjectSimilarityFinder:
    def __init__(self, model_name='all-MiniLM-L6-v2', index_file='subject_index.faiss', subjects_file='subjects.pkl'):
        """
        Initialize the SubjectSimilarityFinder with a sentence transformer model and FAISS index
        
        Args:
            model_name: The name of the sentence transformer model to use
            index_file: Path to save/load the FAISS index
            subjects_file: Path to save/load the list of subjects
        """
        self.model = SentenceTransformer(model_name)
        self.index_file = index_file
        self.subjects_file = subjects_file
        self.subjects = []
        
        # Load existing index and subjects if they exist
        if os.path.exists(index_file) and os.path.exists(subjects_file):
            self.index = faiss.read_index(index_file)
            with open(subjects_file, 'rb') as f:
                self.subjects = pickle.load(f)
        else:
            # Create a new index
            self.index = None
            # Initialize with subjects from the subjects module
            self._initialize_index(get_initial_subjects())
    
    def _initialize_index(self, subjects):
        """
        Initialize the index with the provided subjects
        
        Args:
            subjects: List of subject strings to add
        """
        if not subjects:
            return
        
        # Set the subjects list
        self.subjects = subjects
        
        # Vectorize the subjects
        embeddings = self._vectorize(subjects)
        
        # Create and populate the index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.index.add(embeddings)
        
        # Save the index and subjects
        self._save_index()
        
        print(f"Initialized index with {len(subjects)} subjects.")
    
    def find_similar_subjects(self, query_subject, top_k=3):
        """
        Find the most similar subjects to a query subject
        
        Args:
            query_subject: The subject string to compare against
            top_k: Number of similar subjects to return
            
        Returns:
            List of (subject, similarity_score) tuples
        """
        if not self.subjects:
            return []
        
        # Limit top_k to the number of subjects we have
        top_k = min(top_k, len(self.subjects))
        
        # Vectorize the query
        query_vec = self._vectorize([query_subject])
        
        # Search the index
        scores, indices = self.index.search(query_vec, top_k)
        
        # Get the results, skipping the exact match if present
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            subject = self.subjects[idx]
            # Convert similarity score to percentage (scores are in range [-1, 1])
            similarity_percentage = (float(score) * 100)
            
            # Include all matches, even exact matches
            results.append((subject, similarity_percentage))
        
        # Return top_k results
        return results[:top_k]
    
    def _vectorize(self, subjects):
        """
        Convert subjects to embeddings
        
        Args:
            subjects: List of subject strings
            
        Returns:
            NumPy array of embeddings
        """
        embeddings = self.model.encode(subjects)
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        return embeddings
    
    def _save_index(self):
        """Save the FAISS index and subjects list to disk"""
        faiss.write_index(self.index, self.index_file)
        with open(self.subjects_file, 'wb') as f:
            pickle.dump(self.subjects, f)
    
    def get_all_subjects(self):
        """Return the list of all subjects in the index"""
        return self.subjects


# Example usage
if __name__ == "__main__":
    # Initialize the finder - subjects will be loaded automatically
    finder = SubjectSimilarityFinder()
    
    # Simple loop for searching similar subjects
    while True:
        query = input("\nEnter a subject to find similar ones (or 'exit' to quit): ")
        
        if query.lower() == 'exit':
            print("Goodbye!")
            break
            
        similar = finder.find_similar_subjects(query, top_k=3)
        
        if similar:
            print("\nMost similar subjects:")
            for subj, score in similar:
                print(f"- {subj} ({score:.2f}% similar)")
        else:
            print("No similar subjects found or not enough subjects in the database.")