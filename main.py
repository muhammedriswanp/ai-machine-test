from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq()

croma_client = chromadb.PersistentClient()

collection = croma_client.get_or_create_collection(name="my_collection")


pdf = PyPDFLoader(file_path="Muhammed_Riswan_P (V1).pdf")
pages = pdf.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

chunks = text_splitter.split_text(pages)
# print(chunks)

for i,chunk in enumerate(chunks):
    collection.add(
    ids=[f"id{i}"],
    documents=[chunks],
)
    while True:
        print("Type EXIT to exit")
        user = input("Enter the question")

        if user == "EXIT":
            break

        context = collection.query(
    query_texts=[user]
)

        chat_completion = groq_client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        
        {
            "role": "user",
            "content": f"from this context \n Context={context} answer this question {user}",
        }
    ],

    
    model="openai/gpt-oss-120b"
)


print(chat_completion.choices[0].message.content)


    
 


