from app import config  # loads .env
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL)
print(llm.invoke("Say hello in one sentence.").content)