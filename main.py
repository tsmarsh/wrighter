import os
import argparse
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableMap
from langchain.schema.messages import HumanMessage
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from datetime import datetime
from pathlib import Path
import faiss
import numpy as np
from langchain_openai.embeddings import OpenAIEmbeddings
import logging
import sys

# Configure logging to stderr
logging.basicConfig(
    stream=sys.stderr,  # Direct logs to stderr
    level=logging.INFO,  # Set the log level
    format="%(asctime)s - %(levelname)s - %(message)s"  # Log format
)


# Load environment variables
load_dotenv()

# Load OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please configure it.")

# Initialize FAISS index and metadata storage
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

dimension = 1536  # Dimensions for OpenAI's text-embedding-ada-002
index = faiss.IndexFlatL2(dimension)
metadata_store = {}  # Dictionary to store metadata


# Add to FAISS
def add_to_faiss(role, content):
    vector = np.array(embeddings.embed_query(content)).astype("float32").reshape(1, -1)
    vector_id = index.ntotal  # Use the current total number of vectors as the ID
    index.add(vector)
    metadata_store[vector_id] = {"role": role, "content": content}
    logging.info(f"Added {role} message to FAISS index.")


# Retrieve from FAISS
def retrieve_relevant_chats(query, k=3):
    vector = np.array(embeddings.embed_query(query)).astype("float32").reshape(1, -1)
    distances, indices = index.search(vector, k)
    results = [
        metadata_store[i]
        for i in indices[0] if i != -1 and i in metadata_store
    ]
    return results


# Initialize Rich Console
console = Console()

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Wrighter Chat with Markdown Toggle")
parser.add_argument(
    "--disable-markdown",
    action="store_false",
    help="Start with Markdown rendering disabled."
)
args = parser.parse_args()

# Initialize the LLM using OpenAI Chat Models
llm = ChatOpenAI(
    temperature=0.7,  # Adjust creativity
    model="gpt-4",  # Specify the OpenAI model
)

# Define a prompt template
prompt = PromptTemplate(
    input_variables=["question"],
    template="You are a creative assistant for novel writing. Answer the following question: {question}"
)

# Create a chain using RunnableMap
chain = RunnableMap({
    "question": lambda question: HumanMessage(content=prompt.format(question=question)),
    "output": llm,
})

history_dir = Path("chat_history")


# Save a message to a Markdown file
def save_message(role, content):
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # Create subdirectories for year and month
    subdir = history_dir / datetime.now().strftime("%Y/%m")
    subdir.mkdir(parents=True, exist_ok=True)

    # Define the filename
    filename = subdir / f"{timestamp}-{role}.md"

    # Write content to the Markdown file
    with open(filename, "w") as file:
        file.write(f"---\ntimestamp: {timestamp}\nrole: {role}\n---\n\n")
        file.write(content)

    logging.info(f"Saved {role} message to {filename}")


if __name__ == "__main__":
    console.print("[bold green]Welcome to Wrighter Chat! Type 'exit' to quit.")
    markdown_enabled = args.disable_markdown

    while True:
        user_input = console.input("[bold blue]You:[/bold blue] ")
        if user_input.lower() in {"exit", "quit"}:
            console.print("[bold red]Goodbye![/bold red]")
            break

        if user_input.lower() == "!toggle-markdown":
            markdown_enabled = not markdown_enabled
            status = "enabled" if markdown_enabled else "disabled"
            console.print(f"[bold magenta]Markdown rendering {status}.[/bold magenta]")
            continue

        save_message("user", user_input)
        add_to_faiss("user", user_input)

        # Retrieve relevant past chats
        relevant_chats = retrieve_relevant_chats(user_input, k=3)
        context = "\n".join([f"{chat['role']}: {chat['content']}" for chat in relevant_chats])

        # Combine context with user input
        full_prompt = f"{context}\nUser: {user_input}"
        response = chain.invoke([HumanMessage(full_prompt)])
        txt = response["output"].content

        save_message("assistant", txt)
        add_to_faiss("assistant", txt)

        # Render response based on toggle state
        console.print("[bold yellow]Assistant:[/bold yellow]")
        if markdown_enabled:
            markdown_response = Markdown(txt)
            console.print(markdown_response)
        else:
            console.print(txt)
