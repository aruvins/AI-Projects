from collections import Counter

def build_word_freq(sentences):
    words = []
    for sentence in sentences:
        words.extend(sentence.split())
    word_freq = Counter(words)
    return word_freq

def score_sentences(sentences, word_freq):
    sentence_scores = {}
    for sentence in sentences:
        score = 0
        for word in sentence.split():
            score += word_freq.get(word, 0)
        sentence_scores[sentence] = score
    return sentence_scores

def summarize(sentences, num_sentences=3):
     frequencies = build_word_freq(sentences)
     sentence_scores = score_sentences(sentences, frequencies)

     ranked_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
     top_sentences = [sentence for sentence, score in ranked_sentences[:num_sentences]]
     summary = ' '.join(top_sentences)
     return summary