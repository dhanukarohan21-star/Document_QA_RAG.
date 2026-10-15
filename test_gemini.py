from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

response = llm.invoke(
    "Explain the purpose of a stock market assistant in two sentences."
)

print(response.content)

