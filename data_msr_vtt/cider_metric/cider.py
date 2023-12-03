from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



def cal_cider_score(reference_tokens, candidate_tokens):

    # print('reference_tokens', reference_tokens)
    # print('candidate_tokens', candidate_tokens)

    # Convert tokens to string for TF-IDF vectorization
    generated_text = ' '.join(candidate_tokens)
    reference_texts = [' '.join(reference_tokens)]

    # print('generated_text', generated_text)
    # print('reference_texts', reference_texts)

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([generated_text] + reference_texts)

    # print('type(tfidf_matrix)', type(tfidf_matrix))
    # print('tfidf_matrix:', tfidf_matrix)
    # print('tfidf_matrix[0]:', tfidf_matrix[0])
    # print('tfidf_matrix[1:]:', tfidf_matrix[1:])

    # Calculate cosine similarity
    similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])
    # print('similarities', similarities)

    # Compute consensus score
    consensus_score = similarities.mean()
    # print('consensus_score', consensus_score)

    # Compute sentence length penalty
    generated_length = len(candidate_tokens)
    # reference_lengths = [len(ref) for ref in reference_tokens] # this applies when there are multiple references
    reference_lengths = len(reference_tokens) # this applies when there is only one reference
    # print('generated_length', generated_length)
    # print('reference_lengths', reference_lengths)

    length_penalty = max(0, 1 - abs(generated_length - np.mean(reference_lengths)) / np.mean(reference_lengths))
    # print('length_penalty', length_penalty)

    # Compute CIDEr score
    cider_score = consensus_score * length_penalty
    # print('cider_score', cider_score)
    # stop

    return cider_score

