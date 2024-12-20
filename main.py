import os
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableMap
from langchain.schema.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# Load OpenAI API key from environment variable
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
    print("Welcome to Wrighter Chat! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Generate a response using the chain
        response = chain.invoke([HumanMessage(user_input)])
        print(f"Assistant: {response['output'].content}")