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

# Initialize Rich Console
console = Console()

# Load environment variables
load_dotenv()

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Wrighter Chat with Markdown Toggle")
parser.add_argument(
    "--disable-markdown",
    action="store_false",
    help="Start with Markdown rendering disabled."
)
args = parser.parse_args()

# Load OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please configure it.")

# Initialize the LLM using OpenAI Chat Models
llm = ChatOpenAI(
    temperature=0.7,  # Adjust creativity
    model="gpt-4",    # Specify the OpenAI model
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

# Directory for chat history
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

    print(f"Saved {role} message to {filename}")


if __name__ == "__main__":
    console.print("[bold green]Welcome to Wrighter Chat! Type 'exit' to quit.")
    console.print("[bold cyan]Type '!toggle-markdown' to enable or disable Markdown rendering.[/bold cyan]")
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
        # Generate a response using the chain
        response = chain.invoke([HumanMessage(user_input)])
        txt = response["output"].content
        save_message("assistant", txt)

        # Render response based on toggle state
        console.print("[bold yellow]Assistant:[/bold yellow]")
        if markdown_enabled:
            markdown_response = Markdown(txt)
            console.print(markdown_response)
        else:
            console.print(txt)