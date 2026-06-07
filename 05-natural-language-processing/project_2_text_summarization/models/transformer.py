from transformers import pipeline

# Use explicit task supported in your version
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

def summarize(text):
    result = summarizer(
        text,
        max_new_tokens=80,
        do_sample=False
    )

    return result[0]["summary_text"]