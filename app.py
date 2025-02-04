import os
import json
import uuid
import yaml
import streamlit as st
from flask import Flask, request, jsonify, session
from dotenv import load_dotenv
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain.schema import SystemMessage, HumanMessage
from pyprojroot import here
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import openai

from PIL import Image  # Import PIL Image module
import pytesseract  

import fitz  # PyMuPDF
from unstructured.partition.pdf import partition_pdf

from unstructured.documents.elements import Table, CompositeElement,Image
from unstructured.partition.pdf import partition_pdf
from pathlib import Path





# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Load configuration
config_path = Path(__file__).resolve().parent / "config/tool_config.yml"
with open(config_path,'r') as cnf:
    app_config = yaml.load(cnf, Loader=yaml.FullLoader)



lookup_gst_rag_config = app_config['lookup_gst_rag']
pdf_directory = lookup_gst_rag_config['unstrucutured_data_folder']
vector_db_path = lookup_gst_rag_config['vector_db_path']
collection_name = lookup_gst_rag_config['collection_name']
gst_rag_embedding_model = lookup_gst_rag_config['gst_rag_embedding_model']
k = lookup_gst_rag_config['k']
extract_images_in_pdf = lookup_gst_rag_config['extract_images_in_pdf']
infer_table_structure = lookup_gst_rag_config['infer_table_structure']
chunking_strategy = lookup_gst_rag_config['chunking_strategy']
max_characters = lookup_gst_rag_config['max_characters']
new_after_n_chars = lookup_gst_rag_config['new_after_n_chars']
combine_text_under_n_chars = lookup_gst_rag_config['combine_text_under_n_chars']
llm_tool_rag = lookup_gst_rag_config['llm_tool_rag']




# Initialize LLM
tool_llm = ChatOpenAI(model=llm_tool_rag)




def create_vector_db():
    return Chroma(
        collection_name=collection_name,
        persist_directory=vector_db_path,
        embedding_function=OpenAIEmbeddings()
    )



@tool
def look_gst_data(query: str) -> str:
    """
    Your responsibility is to provide accurate and clear answers related to Goods and Services Tax (GST) in India. 
    You must:
    - Retrieve relevant information from the GST-related documents stored in the database.
    - Ensure your answer is precise and based on actual GST rules, exemptions, or relevant legal content.
    - If the query is complex, provide additional context for better understanding.
    
    Steps:
    1. Query the database for relevant documents related to GST.
    2. Retrieve the top documents (up to 3) based on relevance.
    3. Ensure the response is easy to understand for non-experts.
    4. If the documents don't answer the query, return a message guiding the user to consult a GST professional.

    Example of response structure:
    - Provide the content from the retrieved documents.
    - If necessary, summarize or explain the content in simpler terms.
    """
        
    vectordb = create_vector_db()
    docs = vectordb.similarity_search(query=query, k=3)
    return "\n\n".join([doc.page_content for doc in docs]) if docs else "No relevant GST-related documents found."




@tool
def get_llm_tool(query: str) -> str:
    """
    Treat yourself as an expert in Indian Goods and Services Tax (GST). You should:
    - Solve user queries based on their business needs.
    - Perform calculations if business details with numbers are provided.
    - Provide solid, actionable answers with reasoning and clarity.
    """
    custom_prompt = (
        "You are an expert in Indian Goods and Services Tax (GST). Follow these guidelines to provide **ethical, actionable, and real-world solutions**:\n"
        "1. Read the question carefully and understand the terminology and logic of the query.\n"
        "2. If the query involves a business situation, provide a real-life solution tailored to **Indian GST** to help the user get the most benefit.\n"
        "3. Solve the query based on the user's business needs with **step-by-step actionable advice** that can be easily implemented.\n"
        "4. If the query includes business details with numbers, perform **detailed calculations** based on GST rules and present them clearly.\n"
        "5. Avoid lengthy theoretical explanations. Focus on **applicable GST rules, exemptions, credits, and liabilities** relevant to the user's situation.\n"
        "6. Use reasoning or real-world examples only when necessary to clarify complex points or justify calculations.\n"
        "7. Always deliver a **clear, concise, and implementable answer**.\n"
        "8. Before answering, **recheck** your solution for accuracy and clarity.\n"
        "9. Ensure **ethical** standards are met, including **data privacy** and **non-bias** in all responses.\n"
        "User Query:\n"
        f"{query}\n\n"
        "Your response should be actionable, transparent, and ethically sound, based on Indian GST rules."
    )
    response = tool_llm.invoke(custom_prompt)
    return response.content




class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

tools = [look_gst_data, get_llm_tool]

tool_with_llm = llm.bind_tools(tools=tools)

def chatbot(state: State):
    return {"messages": [tool_with_llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        messages = inputs.get("messages", [])
        if not messages:
            raise ValueError("No message found in input")
        
        last_message = messages[-1]
        outputs = []
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
                outputs.append(ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))
        return {"messages": outputs}

tool_node = BasicToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

def should_continue(state: MessagesState) -> Literal["tools", END]:
    last_message = state['messages'][-1]
    return "tools" if hasattr(last_message, "tool_calls") and last_message.tool_calls else END

graph_builder.add_conditional_edges("chatbot", should_continue, ["tools", END])
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)



def process_pdf(pdf_path):
    print(f"Processing: {pdf_path}")
    # pdf_path  =pdf_path.read()
    raw_pdf_elements = partition_pdf(
        file=pdf_path,
        extract_images_in_pdf=False,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=1000,
        new_after_n_chars=800,
        combine_text_under_n_chars=200,
        
    )

    # Separate text and table elements
    text_elements = []
    table_elements = []
    # image_element = []
    for element in raw_pdf_elements:
        if isinstance(element, Table):
            table_elements.append(str(element))
        elif isinstance(element, CompositeElement):
            text_elements.append(str(element))
        # elif isinstance(element,Image):
        #     image_element.append(str(element))
    


    return "\n".join(text_elements), table_elements  # Return text and tables separately




import numpy as np
import cv2
from ultralytics import YOLO
model = YOLO("yolov8n.pt")


def image_detect_objects(image_file):
    """Detect objects in an image using YOLOv8"""
    image_file.seek(0)  # Reset file pointer to beginning
    image_bytes = np.frombuffer(image_file.read(), np.uint8)
    
    if image_bytes.size == 0:
        return ["Error: Empty image file"]

    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    results = model(image)  # Run YOLO object detection

    detected_objects = []
    for result in results:
        for box in result.boxes:
            detected_objects.append(result.names[int(box.cls[0])])  # Get class name

    return list(set(detected_objects)) if detected_objects else ["No objects detected"]




def analyze_image(image_file):
    """Extract text from an image using Tesseract OCR."""
    image = Image.open(image_file)  # Ensure PIL's Image module is used
    return pytesseract.image_to_string(image)






app = Flask(__name__)
app.secret_key = os.urandom(24)

@app.before_request
def initialize_session_variables():
    if 'chat_history' not in session:
        session['chat_history'] = []
    if 'thread_id' not in session:
        session['thread_id'] = str(uuid.uuid4())


def get_gst_response_from_tool(query):
    config = {"configurable": {"thread_id": session['thread_id']}}
    events = graph.stream({"messages": [("user", query)]}, config, stream_mode="values")
    response = ""
    for event in events:
        # print(event)
        if "messages" in event and event["messages"]:
            message = event["messages"][-1]
            # print(message.content)

            # response += getattr(message, 'pretty_print', lambda: str(message))()

            result = str(message.content) if message is not None else ""
            
            response += result
    return result






@app.route("/UploadData", methods=["POST"])
def UploadData():
    text = request.data.decode('utf-8')
    # print(text,"___")
    pdf_img = request.files.get('file')

    response = ""
    
    if text.strip():
        response = get_gst_response_from_tool(text)
        # print(response)
        session['chat_history'].append(("User", text))
        session['chat_history'].append(("Bot", response))

    if pdf_img:
        file_ext = pdf_img.filename.split('.')[-1].lower()
    
        if file_ext in ["jpg", "jpeg", "png",'webp']:  # Handle image file
            
            text = analyze_image(pdf_img)
            object_text = image_detect_objects(pdf_img)
            both_text = f"this text from the image that have extracted {text} {"\n\n"} and this is the object from images {object_text} name of the image is {pdf_img}"
            response = get_gst_response_from_tool(both_text)

            session['chat_history'].append(("Image Extracted detail", both_text))
            session['chat_history'].append(("Bot", response))
        else:
            text,table = process_pdf(pdf_img)
            final_text = f"this is the text data from the pdf\n\n {text} and this table data from the pdf {table} pdf name is {pdf_img}"
            response = get_gst_response_from_tool(final_text)
        
            session['chat_history'].append(("Pdf Extracted text", final_text))
            session['chat_history'].append(("Bot", response))
    
    return jsonify({"response": response, "chat_history": session['chat_history']}), 200

if __name__ == "__main__":
    app.run(debug=True)
