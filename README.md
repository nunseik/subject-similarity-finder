# Subject Similarity Finder

A Python tool that uses NLP and vector embeddings to find semantically similar subjects from a list of topics.

## Overview

This tool vectorizes a list of course subjects or topics using sentence transformers, stores them in a FAISS index for efficient similarity search, and allows you to find the most similar topics to any input query. The similarity is calculated based on semantic meaning rather than just keyword matching.

## Features

- Uses sentence transformers to create meaningful vector embeddings
- Leverages FAISS for fast vector similarity search
- Returns the top matching subjects along with similarity percentages
- Persists the vector database between runs

## Requirements

- Python 3.7+
- sentence-transformers
- faiss-cpu
- numpy

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/subject-similarity-finder.git
   cd subject-similarity-finder
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv subject-env
   source subject-env/bin/activate  # On Unix/macOS
   # OR
   subject-env\Scripts\activate     # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the script:
   ```bash
   python subject_finder.py
   ```

2. Enter your query subject when prompted:
   ```
   Enter a subject to find similar ones (or 'exit' to quit): machine learning
   ```

3. View results:
   ```
   Most similar subjects:
   - deep-learning (89.42% similar)
   - neural-networks (85.67% similar)
   - artificial-intelligence (82.31% similar)
   ```

## How It Works

The system uses a pre-trained sentence transformer model to convert text subjects into high-dimensional vectors that capture semantic meaning. These vectors are stored in a FAISS index, which enables efficient similarity search even with thousands of subjects.

When you enter a query, it gets converted to a vector using the same model, and then FAISS finds the closest matching vectors in the database, which correspond to the most semantically similar subjects.

## Customizing the Subject List

The subject list is stored in `subjects.py`. You can modify the `get_initial_subjects()` function to add, remove, or update the available subjects.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
