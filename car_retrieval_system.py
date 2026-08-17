"""
TF-IDF Car Information Retrieval System
=========================================
A modular Information Retrieval system built in Python without external ML libraries.
Calculates TF-IDF and Cosine Similarity manually to rank car documents based on user queries.
"""

import os
import sys
import math
import string
import argparse


def parse_car_metadata(file_path):
    """
    Parses metadata fields from a single car document (.txt file).
    Expected fields: Title, Author/Manufacturer, Genre/Car Type, Publication Year/Model Year, Description.
    Returns a dictionary containing parsed metadata and raw text.
    """
    metadata = {
        'file_path': file_path,
        'filename': os.path.basename(file_path),
        'Title': 'Unknown Title',
        'Author': 'Unknown Manufacturer',
        'Genre': 'Unknown Category',
        'Publication Year': 'Unknown Year',
        'Description': '',
        'raw_text': ''
    }

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        metadata['raw_text'] = content

        for line in content.splitlines():
            line_str = line.strip()
            if not line_str or ':' not in line_str:
                continue

            key, value = line_str.split(':', 1)
            key = key.strip().lower()
            value = value.strip()

            if key in ['title', 'car name']:
                metadata['Title'] = value
            elif key in ['author', 'manufacturer', 'brand', 'maker']:
                metadata['Author'] = value
            elif key in ['genre', 'car type', 'category', 'segment']:
                metadata['Genre'] = value
            elif key in ['publication year', 'model year', 'year']:
                metadata['Publication Year'] = value
            elif key in ['description', 'summary']:
                metadata['Description'] = value

    except Exception as e:
        print(f"[Warning] Failed to read or parse file '{file_path}': {e}")

    return metadata


def read_documents(folder_path):
    """
    Reads all .txt files from the specified folder.
    Returns a list of parsed document dictionaries.
    """
    if not os.path.exists(folder_path):
        print(f"[Error] Folder '{folder_path}' does not exist.")
        return []

    documents = []
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    if not files:
        print(f"[Warning] No '.txt' files found in '{folder_path}'.")
        return []

    for filename in sorted(files):
        file_path = os.path.join(folder_path, filename)
        doc_metadata = parse_car_metadata(file_path)
        documents.append(doc_metadata)

    return documents


def preprocess(text):
    """
    Preprocesses input text by:
    1. Converting text to lowercase.
    2. Removing punctuation.
    3. Tokenizing using split().
    Returns a list of cleaned tokens.
    """
    if not text:
        return []

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation by replacing punctuation characters with whitespace
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text_clean = text.translate(translator)

    # Tokenize using split()
    tokens = text_clean.split()
    return tokens


def calculate_tf(tokens):
    """
    Calculates Term Frequency (TF) for each token in a document:
    TF(t) = Number of occurrences of term t / Total number of terms
    Returns a dictionary of term frequencies {term: tf_score}.
    """
    if not tokens:
        return {}

    total_terms = len(tokens)
    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1

    tf_dict = {term: count / total_terms for term, count in counts.items()}
    return tf_dict


def create_vocabulary(documents_tokens):
    """
    Creates a common sorted vocabulary list of all unique terms across all documents.
    """
    vocab = set()
    for tokens in documents_tokens:
        vocab.update(tokens)
    return sorted(list(vocab))


def calculate_idf(documents_tokens, vocabulary):
    """
    Calculates smoothed Inverse Document Frequency (IDF) for each term in vocabulary:
    IDF(t) = log((N + 1) / (DF(t) + 1)) + 1
    where N is the total number of documents and DF(t) is Document Frequency.
    Returns a dictionary of IDF values {term: idf_score}.
    """
    N = len(documents_tokens)
    idf_dict = {}

    for term in vocabulary:
        # Calculate Document Frequency (DF)
        df = sum(1 for doc_tokens in documents_tokens if term in doc_tokens)
        # Smoothed IDF calculation using natural logarithm
        idf_score = math.log((N + 1) / (df + 1)) + 1.0
        idf_dict[term] = idf_score

    return idf_dict


def calculate_tfidf(tf, idf):
    """
    Calculates TF-IDF score for each term:
    TF-IDF(t) = TF(t) * IDF(t)
    Returns a dictionary of TF-IDF values {term: tfidf_score}.
    """
    tfidf_dict = {}
    for term, tf_val in tf.items():
        idf_val = idf.get(term, 0.0)
        tfidf_dict[term] = tf_val * idf_val
    return tfidf_dict


def create_vector(tf_idf_dict, vocabulary):
    """
    Converts a TF-IDF dictionary into a numerical vector corresponding to the vocabulary order.
    Returns a list of float numbers.
    """
    return [tf_idf_dict.get(term, 0.0) for term in vocabulary]


def cosine_similarity(vec1, vec2):
    """
    Calculates Cosine Similarity between two vectors:
    Cosine Similarity(V1, V2) = (V1 . V2) / (||V1|| * ||V2||)
    Handles zero-magnitude vectors safely by returning 0.0.
    """
    if len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(v1 ** 2 for v1 in vec1))
    magnitude2 = math.sqrt(sum(v2 ** 2 for v2 in vec2))

    if magnitude1 == 0.0 or magnitude2 == 0.0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def run_search(query_text, documents, doc_vectors, vocabulary, idf_dict):
    """
    Processes a search query, calculates cosine similarity with all documents,
    ranks results, and displays formatted output.
    """
    print("\n" + "=" * 60)
    print("      TF-IDF CAR INFORMATION RETRIEVAL SYSTEM")
    print("=" * 60)

    print(f"Number of documents loaded : {len(documents)}")
    print(f"Vocabulary size            : {len(vocabulary)}")
    print(f"User Query                 : \"{query_text}\"")

    # Preprocess query
    query_tokens = preprocess(query_text)

    if not query_tokens:
        print("\n[Warning] Query is empty or contains no valid terms after preprocessing.")
        print("Similarity scores for all documents: 0.0000")
        return

    # Identify unknown query words
    vocab_set = set(vocabulary)
    unknown_words = [token for token in query_tokens if token not in vocab_set]

    if unknown_words:
        print(f"Unknown query words        : {', '.join(set(unknown_words))} (Not present in document vocabulary)")
    else:
        print(f"Unknown query words        : None (All query terms recognized)")

    # Compute TF for query
    query_tf = calculate_tf(query_tokens)

    # Compute TF-IDF for query using global IDF values
    query_tfidf = calculate_tfidf(query_tf, idf_dict)

    # Convert query TF-IDF to vector representation
    query_vector = create_vector(query_tfidf, vocabulary)

    # Compute Cosine Similarity against every document vector
    rankings = []
    for idx, doc in enumerate(documents):
        doc_vec = doc_vectors[idx]
        similarity = cosine_similarity(query_vector, doc_vec)
        rankings.append((doc, similarity))

    # Rank documents by similarity score (descending)
    rankings.sort(key=lambda x: x[1], reverse=True)

    # Display Complete Document Ranking
    print("\n" + "-" * 60)
    print("COMPLETE DOCUMENT RANKING")
    print("-" * 60)
    print(f"{'Rank':<6}{'Score':<10}{'Car Title':<30}{'Category':<15}")
    print("-" * 60)
    for rank, (doc, score) in enumerate(rankings, 1):
        print(f"{rank:<6}{score:<10.4f}{doc['Title']:<30}{doc['Genre']:<15}")

    # Display Top 3 Most Relevant Cars
    top_n = min(3, len(rankings))
    print("\n" + "=" * 60)
    print(f"TOP {top_n} MOST RELEVANT CARS")
    print("=" * 60)

    for rank, (doc, score) in enumerate(rankings[:top_n], 1):
        print(f"\n--- Rank {rank} [Relevance Score: {score:.4f}] ---")
        print(f"  Car Name        : {doc['Title']}")
        print(f"  Manufacturer    : {doc['Author']}")
        print(f"  Car Type        : {doc['Genre']}")
        print(f"  Model Year      : {doc['Publication Year']}")
        print(f"  Description     : {doc['Description']}")

    print("\n" + "=" * 60 + "\n")


def main():
    """
    Main entry point for the TF-IDF Car Information Retrieval System.
    Supports both command-line arguments (--query) and interactive prompt search.
    """
    parser = argparse.ArgumentParser(description="TF-IDF Car Information Retrieval System")
    parser.add_argument("--query", "-q", type=str, help="Search query string")
    parser.add_argument("--folder", "-f", type=str, default="documents", help="Path to documents folder")
    args = parser.parse_args()

    folder_path = args.folder

    # Step 1: Read documents from documents folder
    print(f"Loading documents from '{folder_path}'...")
    documents = read_documents(folder_path)

    if not documents:
        print("[Error] System cannot proceed without car documents. Please create a 'documents' folder with .txt files.")
        return

    # Step 2: Preprocess document text
    documents_tokens = []
    for doc in documents:
        tokens = preprocess(doc['raw_text'])
        documents_tokens.append(tokens)

    # Step 3: Create global vocabulary
    vocabulary = create_vocabulary(documents_tokens)

    if not vocabulary:
        print("[Error] Vocabulary is empty. All documents appear to be empty or contain no words.")
        return

    # Step 4: Calculate IDF for vocabulary
    idf_dict = calculate_idf(documents_tokens, vocabulary)

    # Step 5: Convert documents into TF-IDF vectors
    doc_vectors = []
    for tokens in documents_tokens:
        tf_dict = calculate_tf(tokens)
        tfidf_dict = calculate_tfidf(tf_dict, idf_dict)
        vector = create_vector(tfidf_dict, vocabulary)
        doc_vectors.append(vector)

    # Step 6: Process Query (CLI or Interactive)
    if args.query:
        run_search(args.query, documents, doc_vectors, vocabulary, idf_dict)
    else:
        # Interactive Search Loop
        print("\nWelcome to the TF-IDF Car Information Retrieval System!")
        print("Type your search query below, or type 'exit' / 'quit' to close.")

        while True:
            try:
                user_query = input("\nEnter car search query: ").strip()
                if user_query.lower() in ['exit', 'quit']:
                    print("Exiting search system. Goodbye!")
                    break
                if not user_query:
                    print("[Warning] Please enter a non-empty query.")
                    continue

                run_search(user_query, documents, doc_vectors, vocabulary, idf_dict)

            except (KeyboardInterrupt, EOFError):
                print("\nExiting search system. Goodbye!")
                break


if __name__ == "__main__":
    main()
