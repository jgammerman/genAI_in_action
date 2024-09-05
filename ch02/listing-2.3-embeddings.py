import os
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI()

def get_embedding(text):
    # Request an embedding from the OpenAI API
    response = client.embeddings.create(
        model="text-embedding-ada-002",  # Use the 'text-embedding-ada-002' model
        input=text  # The text to generate an embedding for
    )
    # Return the embedding vector from the response
    return response.data[0].embedding

# Get an embedding for a sample text
embeddings = get_embedding("I have a white dog named Champ.")
# Print the length of the embedding vector
print("Embedding Length:", len(embeddings))
# Print the first five elements of the embedding vector
print("Embedding:", embeddings[:5])
