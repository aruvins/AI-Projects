def chunk_text(text, chunk_size=128):
    """
    Splits the input text into chunks of specified size
    It splits the text into words and then groups them into chunks of the specified size.
    Chunking is useful for processing large texts that may exceed the input limits of certain models or algorithms.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum size of each chunk. Default is 128.

    Returns:
        List[str]: A list of text chunks.
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks