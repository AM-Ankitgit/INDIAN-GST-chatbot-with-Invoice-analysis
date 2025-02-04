import os
import sys
import uuid
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, CompositeElement
from pyprojroot import here
from dotenv import load_dotenv
import yaml



def get_vectors():
    embedding_function = OpenAIEmbeddings()
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=vector_db_path,
        embedding_function=embedding_function
    )
    return vectorstore

# Function to extract and process data from a single PDF
def process_pdf(pdf_path):
    raw_pdf_elements = partition_pdf(
        filename=pdf_path,
        extract_images_in_pdf=extract_images_in_pdf,
        infer_table_structure=infer_table_structure,
        chunking_strategy=chunking_strategy,
        max_characters=max_characters,
        new_after_n_chars=new_after_n_chars,
        combine_text_under_n_chars=combine_text_under_n_chars
        )

    # Separate text and table elements
    text_elements = []
    table_elements = []
    for element in raw_pdf_elements:
        if isinstance(element, Table):
            table_elements.append(str(element))
        elif isinstance(element, CompositeElement):
            text_elements.append(str(element))
    
    return text_elements, table_elements

# Batch process all PDFs in the directory

def PrepareVectorDB(pdf_directory):
    try:
        all_files = os.listdir(pdf_directory)[:1]
        vectorstore = get_vectors()
        for pdf_file in all_files:
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(pdf_directory, pdf_file)
                text_elements, table_elements = process_pdf(pdf_path)
                # print(text_elements,table_elements)

                # Generate unique IDs for each document
                text_doc_ids = [str(uuid.uuid4()) for _ in text_elements]
                table_doc_ids = [str(uuid.uuid4()) for _ in table_elements]

                # Prepare text and table documents
                text_documents = [
                    Document(page_content=text, metadata={"type": "text", "source": pdf_file, "doc_id": doc_id})
                    for text, doc_id in zip(text_elements, text_doc_ids)
                ]
                table_documents = [
                    Document(page_content=table, metadata={"type": "table", "source": pdf_file, "doc_id": doc_id})
                    for table, doc_id in zip(table_elements, table_doc_ids)
                ]

                # Add documents to ChromaDB
                # print(table_documents)

                if text_documents:
                    vectorstore.add_documents(text_documents)
                if table_documents:
                    vectorstore.add_documents(table_documents)
        vectorstore.persist()
        print("All PDFs have been processed and stored in ChromaDB.")
    except Exception as e:
        print(e)


if __name__=="__main__":

    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


    import yaml
    from pathlib import Path

    config_path = Path(__file__).resolve().parent / "config/tool_config.yml"
  
    with open(config_path, "r") as cnf:
        app_config = yaml.load(cnf, Loader=yaml.FullLoader)

    lookup_gst_rag_config = app_config['lookup_gst_rag']
    pdf_directory = lookup_gst_rag_config['unstrucutured_data_folder']
    vector_db_path= lookup_gst_rag_config['vector_db_path']
    collection_name = lookup_gst_rag_config['collection_name']
    gst_rag_embedding_model = lookup_gst_rag_config['gst_rag_embedding_model']
    k            = lookup_gst_rag_config['k']
    extract_images_in_pdf = lookup_gst_rag_config['extract_images_in_pdf']
    infer_table_structure = lookup_gst_rag_config['infer_table_structure']
    chunking_strategy = lookup_gst_rag_config['chunking_strategy']
    max_characters = lookup_gst_rag_config['max_characters']
    new_after_n_chars = lookup_gst_rag_config['new_after_n_chars']
    combine_text_under_n_chars = lookup_gst_rag_config['combine_text_under_n_chars']
    llm_tool_rag = lookup_gst_rag_config['llm_tool_rag']

    
    PrepareVectorDB(pdf_directory)



