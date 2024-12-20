import os
import argparse
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableMap
from langchain.schema.messages import HumanMessage
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
import logging
import sys

# Configure logging to stderr
from chat_history import save_message
from search import add_to_search, retrieve_relevant_chats

load_dotenv()

logging.basicConfig(
    stream=sys.stderr,  # Direct logs to stderr
    level=logging.INFO,  # Set the log level
    format="%(asctime)s - %(levelname)s - %(message)s"  # Log format
)


openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please configure it.")


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
        add_to_search("user", user_input)

        # Retrieve relevant past chats
        relevant_chats = retrieve_relevant_chats(user_input, k=3)
        context = "\n".join([f"{chat['role']}: {chat['content']}" for chat in relevant_chats])

        # Combine context with user input
        full_prompt = f"{context}\nUser: {user_input}"
        response = chain.invoke([HumanMessage(full_prompt)])
        txt = response["output"].content

        save_message("assistant", txt)
        add_to_search("assistant", txt)

        # Render response based on toggle state
        console.print("[bold yellow]Assistant:[/bold yellow]")
        if markdown_enabled:
            markdown_response = Markdown(txt)
            console.print(markdown_response)
        else:
            console.print(txt)
