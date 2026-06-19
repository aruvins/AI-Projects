def chunk_text(text):
    """
    Splits the input text into chunks of specified size
    It splits the text into words and then groups them into chunks of the specified size.
    Chunking is useful for processing large texts that may exceed the input limits of certain models or algorithms.

    Args:
        text (str): The input text to be chunked.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

    return chunks