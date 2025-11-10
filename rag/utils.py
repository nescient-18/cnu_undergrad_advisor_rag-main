from collections.abc import Sized, Mapping, Iterable
from typing import Any

def get_information_about_this_object(some_object: Any) -> dict:
    """
    Get information about the given object.

    Args:
        some_object (Any): The object to inspect.

    Returns:
        dict: A dictionary containing information about the object, including its type, length (if applicable),
              whether it is a mapping or iterable, keys and values (if it is a mapping), attributes, and information
              about the first element (if it is an iterable).
    """
    def get_attributes(obj):
        """
        Returns a dictionary containing the attributes of the given object.

        Parameters:
        obj (object): The object to inspect.

        Returns:
        dict: A dictionary where the keys are the attribute names and the values are the attribute values.

        """
        attributes = {attr: getattr(obj, attr) for attr in dir(obj) 
                      if not attr.startswith('__') and not callable(getattr(obj, attr))}
        for attr, value in attributes.items():
            if isinstance(value, str):
                if len(value) > 100:
                    value = "#### TRUNCATED AT 10% #### " + value[:100]
                attributes[attr] = value
        return attributes

    info = {
        "type_of_object": type(some_object),
        "length_of_object": len(some_object) if isinstance(some_object, Sized) else None,
        "is_mapping": isinstance(some_object, Mapping),
        "is_iterable": isinstance(some_object, Iterable),
        "keys_of_object": list(some_object.keys()) if isinstance(some_object, Mapping) else None,
        "values_of_object": list(some_object.values()) if isinstance(some_object, Mapping) else None,
        "items_of_object": list(some_object.items()) if isinstance(some_object, Mapping) else None,
        "attributes_of_object": get_attributes(some_object),
        # "first_element": None,
        "type_of_first_element": None,
        "length_of_first_element": None,
        "attributes_of_first_element": None
    }

    if info["is_iterable"]:
        try:
            first_element = next(iter(some_object))
            info["type_of_first_element"] = type(first_element)
            info["length_of_first_element"] = len(first_element) if isinstance(first_element, Sized) else None
            info["attributes_of_first_element"] = get_attributes(first_element)
            # info["first_element"] = first_element
        except StopIteration:
            pass  # The iterable is empty

    return info

def print_object_info(some_object: Any):
    """
    Prints information about the given object.

    Args:
        some_object (Any): The object to inspect.

    Returns:
        None
    """
    info = get_information_about_this_object(some_object)
    for key, value in info.items():
        if isinstance(value, str) and len(value) > 100:
            value = value[:int(len(value) * 0.1)]  # Take only 10% of the string
        if key in ["attributes_of_object", "attributes_of_first_element"]:
            print(f"{key}:")
            try:
                for attr, attr_value in value.items():
                    print(f"  {attr}: {attr_value}")
            except:
                pass
        else:
            print(f"{key}: {value}")
        print(f"-" * 50)

def get_largest_tokens(list_of_docs, tokenizer_name="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    """
    Get the largest number of tokens in a document from a list of documents.

    Args:
    list_of_docs (list): A list of documents.
    tokenizer_name (str): The name of the tokenizer to use.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    largest_token_count = 0
    for document in list_of_docs:
        tokens = tokenizer.encode(document.page_content)
        num_tokens = len(tokens)
        if num_tokens > largest_token_count:
            largest_token_count = num_tokens

    print(f"Largest token count: {largest_token_count}")

def get_tokens(text, tokenizer_name):
    """
    Get the number of tokens in a string.

    Args:
    text (str): The text to tokenize.
    tokenizer_name (str): The name of the tokenizer to use.
    """
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Tokenize the string
    tokens = tokenizer.encode(text)

    # Count the number of tokens
    num_tokens = len(tokens)

    print(f"Number of tokens: {num_tokens}")