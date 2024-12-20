import faiss
import logging
from sentence_transformers import SentenceTransformer

local_embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')

dimension = 384  # Dimensions for OpenAI's text-embedding-ada-002
index = faiss.IndexFlatL2(dimension)
metadata_store = {}  # Dictionary to store metadata


def calculate_local_embedding(content):
    return local_embeddings_model.encode(content, convert_to_numpy=True)


def add_to_search(role, content):
    vector = calculate_local_embedding(content).astype("float32").reshape(1, -1)
    vector_id = index.ntotal  # Use the current total number of vectors as the ID
    index.add(vector)
    metadata_store[vector_id] = {"role": role, "content": content}
    logging.info(f"Added {role} message to FAISS index.")


def retrieve_relevant_chats(query, k=3):
    vector = calculate_local_embedding(query).astype("float32").reshape(1, -1)
    distances, indices = index.search(vector, k)
    results = [
        metadata_store[i]
        for i in indices[0] if i != -1 and i in metadata_store
    ]
    logging.info(f"Relevant chats: {results}")

    return results


def load_files(files):
    for chat_file in files:
        try:
            with open(chat_file, "r") as file:
                lines = file.readlines()
                timestamp = lines[1].split(": ", 1)[1].strip()
                role = lines[2].split(": ", 1)[1].strip()
                content = "".join(lines[4:]).strip()

                # Calculate embedding and add to FAISS
                vector = calculate_local_embedding(content).astype("float32").reshape(1, -1)
                vector_id = index.ntotal  # Use current total as the ID
                index.add(vector)
                metadata_store[vector_id] = {"timestamp": timestamp, "role": role, "content": content}
                logging.info(f"Loaded {role} message from {chat_file} into FAISS index.")
        except Exception as e:
            logging.error(f"Error loading {chat_file}: {e}")