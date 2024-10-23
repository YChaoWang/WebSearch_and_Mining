import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import numpy as np

# Check if the required resources are available
try:
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")

# Verify the NLTK data path
# print("NLTK data paths:")
# print(nltk.data.path)


def extract_nouns_verbs(text):
    """Extract nouns and verbs from the given text."""
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    # Extract nouns and verbs
    nouns_verbs = [
        word for word, tag in tagged if tag.startswith("NN") or tag.startswith("VB")
    ]
    return nouns_verbs


def create_new_query_vector(original_query, feedback_terms, vector_index):
    """Create a new query vector based on the original query and feedback terms."""
    # Initialize the original query vector
    original_vector = np.zeros(len(vector_index))

    # Set weights for the original query
    for term in original_query.split():
        if term in vector_index:
            original_vector[vector_index[term]] += 1

    # Initialize the feedback vector
    feedback_vector = np.zeros(len(vector_index))

    # Set weights for the feedback terms
    for term in feedback_terms:
        if term in vector_index:
            feedback_vector[
                vector_index[term]
            ] += 0.5  # Weight of 0.5 for feedback terms

    # Combine the vectors
    new_query_vector = original_vector + feedback_vector
    return new_query_vector


def pseudo_feedback(vector_space, original_query, documents, file_paths):
    """Perform pseudo feedback to re-rank documents based on the original query."""
    # Step 1: Perform initial search
    initial_results = vector_space.search(
        original_query.split(),
        method="cosine",
        weighting="tf-idf",
        file_paths=file_paths,
    )

    # Debugging output
    # print(f"Initial results: {initial_results}")
    print(f"Number of documents: {len(documents)}")

    # Step 2: Extract Nouns and Verbs from the first document
    if initial_results:
        first_doc_id = initial_results[0][0]  # Get the ID of the first document
        if first_doc_id >= len(documents):
            print(f"Error: first_doc_id {first_doc_id} is out of range for documents.")
            return []

        feedback_document = documents[
            first_doc_id
        ]  # Retrieve the content of the first document
        feedback_terms = extract_nouns_verbs(
            feedback_document
        )  # Extract nouns and verbs

        # Debugging output for feedback terms
        print(f"Feedback terms extracted: {feedback_terms}")

        # Ensure feedback_terms are strings
        feedback_terms = [str(term) for term in feedback_terms]

        # Combine feedback terms into a new query string
        new_query = " ".join(feedback_terms)
        print(f"New query created from feedback terms: {new_query}")

        # Step 3: Create a new query vector
        # vector_index = (
        #     vector_space.vectorKeywordIndex
        # )  # Use the vector index from VectorSpace
        # new_query_vector = create_new_query_vector(
        #     new_query, feedback_terms, vector_index
        # )

        # Step 4: Re-rank documents using the new query vector
        print("Re-ranking documents using the new query vector...")
        re_ranked_results = vector_space.search(
            new_query.split(),  # Use the new query string split into terms
            method="cosine",
            weighting="tf-idf",
            file_paths=file_paths,
        )

        return re_ranked_results
    else:
        print("No initial results found.")
        return []


# Example usage
# Assuming you have a VectorSpace instance and documents loaded
# vector_space = VectorSpace(eng_documents, language="english")
# original_query = "network"
# re_ranked_results = pseudo_feedback(vector_space, original_query, eng_documents)
# print(re_ranked_results)
