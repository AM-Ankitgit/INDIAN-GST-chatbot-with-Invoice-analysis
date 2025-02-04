from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# GST tool
@tool
def look_gst_data(query: str) -> str:
    """
    your full resonsible to answere the Goods and Services Tax (INDIA) present in documents
    """
    vectordb = Chroma(
        collection_name="rag-gst",
        persist_directory="./DATA",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    )
    docs = vectordb.similarity_search(query=query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])

# LLM tool
@tool
def get_llm_tool(query: str) -> str:
    """
    you are capable to solve any mathematical operation
    """
    response = ChatOpenAI(model="o1-mini").invoke(query)
    return response.content

tools = [look_gst_data, get_llm_tool]
tool_with_llm = llm.bind_tools(tools=tools)

# Final invocation
response = tool_with_llm.invoke("Select the number from among the given options that can replace the question mark (?) in the following series. 28, 32, 41, 57 ?.## i want one line answere")
print(response)
