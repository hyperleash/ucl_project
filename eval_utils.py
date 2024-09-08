def get_words_from_ids(id_list, word_to_id_dict):
    id_to_word_dict = {v: k for k, v in word_to_id_dict.items()}  # Reverse the dictionary
    return [id_to_word_dict[id] for id in id_list]
    
    # Define a function to extract context around a token
def get_context_around_token(token_idx, tokens, word_to_id_dict, window_size=2):
    """Extracts a context window of words around a given token index."""
    start = max(0, token_idx - window_size)
    end = min(len(tokens), token_idx + window_size + 1)  # +1 to include the token itself
    return get_words_from_ids(tokens[start:end], word_to_id_dict=word_to_id_dict)