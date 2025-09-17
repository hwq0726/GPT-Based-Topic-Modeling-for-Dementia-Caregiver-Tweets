"""
GPT Topic Coherence Calculator

This module provides functions to calculate topic coherence scores for topics extracted by GPT,
where topics are represented as lists of words rather than word-frequency tuples.

The coherence calculation uses Gensim's CoherenceModel which requires:
1. Topics: List of word lists (one list per topic)
2. Texts: Tokenized documents 
3. Dictionary: Gensim dictionary object
4. Coherence type: 'c_v', 'u_mass', 'c_uci', 'c_npmi'

Author: Generated for GPT topic modeling evaluation
"""

import pickle
import numpy as np
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary


def gpt_topic_coherence(gpt_topics, texts, dictionary, coherence_type='c_v'):
    """
    Calculate topic coherence for GPT-extracted topics.
    
    Parameters:
    -----------
    gpt_topics : list of lists
        List where each element is a list of words representing a topic.
        Example: [['word1', 'word2', 'word3'], ['word4', 'word5', 'word6']]
    
    texts : list of lists
        Tokenized documents. Each document is a list of tokens.
        Example: [['token1', 'token2'], ['token3', 'token4']]
    
    dictionary : gensim.corpora.Dictionary
        Gensim dictionary object created from the texts
    
    coherence_type : str, optional
        Type of coherence measure to use. Options:
        - 'c_v': C_V coherence (default)
        - 'u_mass': U_mass coherence
        - 'c_uci': C_UCI coherence
        - 'c_npmi': C_NPMI coherence
    
    Returns:
    --------
    tuple: (coherence_scores_topics, coherence_score_model)
        - coherence_scores_topics: List of coherence scores for each individual topic
        - coherence_score_model: Overall coherence score for the entire model
    """
    print(f"Topic coherence type: {coherence_type}")
    print(f"Number of topics: {len(gpt_topics)}")
    
    coherence_scores_topics = []
    words_topics = []
    
    # Calculate coherence for each individual topic
    for i, topic_words in enumerate(gpt_topics):
        print(f"Calculating coherence for topic {i}: {topic_words[:5]}...")  # Show first 5 words
        
        # Create CoherenceModel for this single topic
        cm = CoherenceModel(
            topics=[topic_words], 
            texts=texts, 
            dictionary=dictionary, 
            coherence=coherence_type, 
            processes=1
        )
        coherence = cm.get_coherence()
        coherence_scores_topics.append(coherence)
        words_topics.append(topic_words)
    
    # Calculate overall model coherence
    print("Calculating overall model coherence...")
    coherence_score_model = CoherenceModel(
        topics=words_topics, 
        texts=texts, 
        dictionary=dictionary, 
        coherence=coherence_type, 
        processes=1
    ).get_coherence()
    
    return coherence_scores_topics, coherence_score_model


def load_texts_and_dictionary(texts_path, dictionary_path):
    """
    Load tokenized texts and dictionary from pickle files.
    
    Parameters:
    -----------
    texts_path : str
        Path to pickle file containing tokenized texts
    
    dictionary_path : str
        Path to pickle file containing gensim dictionary
    
    Returns:
    --------
    tuple: (texts, dictionary)
        - texts: List of tokenized documents
        - dictionary: Gensim dictionary object
    """
    # print(f"Loading texts from: {texts_path}")
    with open(texts_path, 'rb') as f:
        texts = pickle.load(f)
    
    # print(f"Loading dictionary from: {dictionary_path}")
    with open(dictionary_path, 'rb') as f:
        dictionary = pickle.load(f)
    
    # print(f"Loaded {len(texts)} documents")
    # print(f"Dictionary contains {len(dictionary)} unique tokens")
    
    return texts, dictionary


def create_dictionary_from_texts(texts, min_count=5, max_freq=0.5):
    """
    Create a gensim dictionary from tokenized texts.
    
    Parameters:
    -----------
    texts : list of lists
        Tokenized documents
    
    min_count : int, optional
        Minimum word count to include in dictionary (default: 5)
    
    max_freq : float, optional
        Maximum word frequency to include in dictionary (default: 0.5)
    
    Returns:
    --------
    gensim.corpora.Dictionary
        Gensim dictionary object
    """
    print(f"Creating dictionary with min_count={min_count}, max_freq={max_freq}")
    dictionary = Dictionary(texts)
    
    # Filter extremes
    dictionary.filter_extremes(no_below=min_count, no_above=max_freq)
    
    print(f"Dictionary created with {len(dictionary)} unique tokens")
    return dictionary


def evaluate_gpt_topics(gpt_topics, texts, dictionary, coherence_types=['c_v', 'u_mass']):
    """
    Evaluate GPT topics using multiple coherence measures.
    
    Parameters:
    -----------
    gpt_topics : list of lists
        GPT-extracted topics
    
    texts : list of lists
        Tokenized documents
    
    dictionary : gensim.corpora.Dictionary
        Gensim dictionary
    
    coherence_types : list of str
        List of coherence measures to calculate
    
    Returns:
    --------
    dict
        Dictionary with coherence results for each measure
    """
    results = {}
    
    for coherence_type in coherence_types:
        # print(f"\n{'='*50}")
        # print(f"Evaluating with {coherence_type} coherence")
        # print(f"{'='*50}")
        
        topic_scores, model_score = gpt_topic_coherence(
            gpt_topics, texts, dictionary, coherence_type
        )
        
        results[coherence_type] = {
            'topic_scores': topic_scores,
            'model_score': model_score,
            'mean_topic_score': np.mean(topic_scores),
            'std_topic_score': np.std(topic_scores)
        }
        
        # print(f"\nResults for {coherence_type}:")
        # print(f"Model coherence score: {model_score:.4f}")
        # print(f"Mean topic coherence: {np.mean(topic_scores):.4f} ± {np.std(topic_scores):.4f}")
        # print(f"Individual topic scores: {[f'{score:.4f}' for score in topic_scores]}")
    
    return results


# Example usage and testing functions
def example_usage():
    """
    Example of how to use the GPT topic coherence functions.
    """
    print("GPT Topic Coherence Calculator - Example Usage")
    print("=" * 50)
    
    # Example GPT topics (replace with your actual GPT results)
    example_gpt_topics = [
        ['health', 'medical', 'doctor', 'patient', 'treatment', 'care', 'hospital'],
        ['technology', 'computer', 'software', 'programming', 'data', 'algorithm'],
        ['sports', 'football', 'basketball', 'game', 'player', 'team', 'match'],
        ['food', 'restaurant', 'cooking', 'recipe', 'delicious', 'taste', 'meal'],
        ['travel', 'vacation', 'trip', 'hotel', 'flight', 'destination', 'journey']
    ]
    
    print("Example GPT topics:")
    for i, topic in enumerate(example_gpt_topics):
        print(f"Topic {i}: {topic}")
    
    # Load your actual data (same as used in original pipeline)
    texts, dictionary = load_texts_and_dictionary(
        'baseline/Twitter_Text_Data/positives_tokens.pkl',
        'baseline/Twitter_Text_Data/positives_dictionary.pkl'
    )
    
    # Evaluate topics using your actual data
    results = evaluate_gpt_topics(example_gpt_topics, texts, dictionary)
    
    return results


if __name__ == "__main__":
    # Run example when script is executed directly
    results = example_usage()
    
    print("\n" + "="*50)
    print("SUMMARY OF RESULTS")
    print("="*50)
    
    for coherence_type, result in results.items():
        print(f"\n{coherence_type.upper()} Coherence:")
        print(f"  Model Score: {result['model_score']:.4f}")
        print(f"  Mean Topic Score: {result['mean_topic_score']:.4f} ± {result['std_topic_score']:.4f}")
