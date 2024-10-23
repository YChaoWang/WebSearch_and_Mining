import math
from collections import defaultdict
import os
from tqdm import tqdm
from Parser import Parser
import util


class VectorSpace:
    """A simplified vector space model for document search using TF-IDF."""

    def __init__(self, documents=[]):
        self.documentTfVectors = []
        self.documentIdfVectors = []
        self.documentTfIdfVectors = []
        self.vectorKeywordIndex = {}
        self.parser = Parser()
        if len(documents) > 0:
            self.build(documents)

    def build(self, documents):
        """Create the vector space for the passed document strings"""
        print("Building vector space...")
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)
        print(f"Vocabulary size: {len(self.vectorKeywordIndex)}")

        print("Creating TF vectors...")
        self.documentTfVectors = [
            self.makeTfVector(document)
            for document in tqdm(documents, desc="Processing documents")
        ]

        print("Creating IDF vectors...")
        self.documentIdfVectors = self.makeIdfVectors(documents)

        print("Creating TF-IDF vectors...")
        self.documentTfIdfVectors = self.makeTfIdfVectors()

    def getVectorKeywordIndex(self, documentList):
        """Create the keyword to vector index mapping"""
        vocabularyString = " ".join(documentList)
        vocabularyList = self.parser.tokenise(vocabularyString)
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = sorted(set(vocabularyList))

        vectorIndex = {word: offset for offset, word in enumerate(uniqueVocabularyList)}
        return vectorIndex

    def makeTfVector(self, wordString):
        """Create the TF vector for a document"""
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)

        for word in wordList:
            if word in self.vectorKeywordIndex:  # 確保詞在索引中
                vector[self.vectorKeywordIndex[word]] += 1

        return vector

    def build_word_doc_count(self, documents):
        """Count document frequency for each word"""
        word_doc_count = defaultdict(int)

        for document in documents:
            wordList = self.parser.tokenise(document)
            wordList = self.parser.removeStopWords(wordList)
            uniqueWords = set(wordList)

            for word in uniqueWords:
                word_doc_count[word] += 1

        return word_doc_count

    def makeIdfVectors(self, documents):
        """Create IDF vector for all terms"""
        word_doc_count = self.build_word_doc_count(documents)
        total_documents = len(documents)

        idfVector = [0] * len(self.vectorKeywordIndex)

        for term, index in tqdm(
            self.vectorKeywordIndex.items(), desc="Calculating document IDF"
        ):
            docs_with_term = word_doc_count.get(term, 0)
            idf = math.log(total_documents / (1 + docs_with_term)) + 1
            idfVector[index] = idf

        return idfVector

    def makeTfIdfVectors(self):
        """Create TF-IDF vectors by combining TF and IDF"""
        tfIdfVectors = []
        for tfVector in tqdm(
            self.documentTfVectors, desc="Creating document TF-IDF vectors"
        ):
            tfIdfVector = [
                tf * idf for tf, idf in zip(tfVector, self.documentIdfVectors)
            ]
            tfIdfVectors.append(tfIdfVector)

        return tfIdfVectors

    def buildQueryVector(self, termList):
        """Convert query string into a vector"""
        queryTfVector = self.makeTfVector(" ".join(termList))

        queryTfIdfVector = [
            tf * idf for tf, idf in zip(queryTfVector, self.documentIdfVectors)
        ]

        return queryTfIdfVector

    def euclidean_distance(self, vector1, vector2):
        """Calculate the Euclidean distance between two vectors"""
        if len(vector1) != len(vector2):
            raise ValueError("Vectors must be of equal length")

        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vector1, vector2)))

    def related(self, documentId, method="cosine"):
        """Find related documents to the given document ID"""
        ratings = []

        for documentVector in self.documentTfIdfVectors:
            if method == "cosine":
                rating = util.cosine(
                    self.documentTfIdfVectors[documentId], documentVector
                )
            else:  # euclidean
                rating = self.euclidean_distance(
                    self.documentTfIdfVectors[documentId], documentVector
                )
            ratings.append(rating)

        return ratings

    def search(self, searchList, method="cosine", weighting="tf-idf", file_paths=[]):
        """Search for documents that match based on a list of terms"""
        if weighting == "tf-idf":
            queryVector = self.buildQueryVector(searchList)
            documentVectors = self.documentTfIdfVectors
        else:  # Raw TF weighting
            queryVector = self.makeTfVector(" ".join(searchList))
            documentVectors = self.documentTfVectors

        if method == "cosine":
            print(f"Calculating Cosine similarity with {weighting} weighting...")
            ratings = [
                util.cosine(queryVector, documentVector)
                for documentVector in tqdm(documentVectors, desc="Processing documents")
            ]
        else:  # Euclidean distance
            print(f"Calculating Euclidean distance with {weighting} weighting...")
            ratings = [
                self.euclidean_distance(queryVector, documentVector)
                for documentVector in tqdm(documentVectors, desc="Processing documents")
            ]

        print("\nNewsID Score")
        indexed_ratings = list(enumerate(ratings))
        top_ratings = sorted(
            indexed_ratings, key=lambda x: x[1], reverse=(method == "cosine")
        )[:10]

        for index, score in top_ratings:
            file_name = os.path.basename(file_paths[index])
            print(f"{file_name}  {score:.7f}")

        return top_ratings
