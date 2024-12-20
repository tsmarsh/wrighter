import os
import argparse
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableMap
from langchain.schema.messages import HumanMessage
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

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

        # Generate a response using the chain
        response = chain.invoke([HumanMessage(user_input)])

        # Render response based on toggle state
        console.print("[bold yellow]Assistant:[/bold yellow]")
        if markdown_enabled:
            markdown_response = Markdown(response["output"].content)
            console.print(markdown_response)
        else:
            console.print(response["output"].content)