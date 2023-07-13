import os
import getpass

from langchain import document_loaders
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake

os.environ["ACTIVELOOP_TOKEN"] = "eyJhbGciOiJIUzUxMiIsImlhdCI6MTY4OTI1OTkxNiwiZXhwIjoxNzIwODgyMjU5fQ.eyJpZCI6ImlpYXNjZXJpIn0.Fyryv5DagPWTN86DDXmkFmCibkcmiyJ0B21pN5-CpMRzZZZ4vVEnY-zo7SD6F9B9X4K6KLFWOWGVkLJQvtHXBA"
os.environ["OPENAI_API_KEY"] = "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"
embeddings = OpenAIEmbeddings(disallowed_special=())

# Load all files inside the repository
import os
from langchain.document_loaders import TextLoader

root_dir = './jwlibrary'
docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        try:
            loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
            docs.extend(loader.load_and_split())
        except Exception as e:
            pass

# Chunk the files

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

import deeplake
username = "iiasceri" # replace with your username from app.activeloop.ai
db = DeepLake(dataset_path=f"hub://{username}/jwlibrary", embedding_function=embeddings) #dataset would be publicly available
db.add_documents(texts)

db = DeepLake(dataset_path=f"hub://{username}/motion-canvas", read_only=True, embedding_function=embeddings)
retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['fetch_k'] = 100
retriever.search_kwargs['maximal_marginal_relevance'] = True
retriever.search_kwargs['k'] = 10

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

model = ChatOpenAI(model='gpt-3.5-turbo') # switch to 'gpt-4'
qa = ConversationalRetrievalChain.from_llm(model,retriever=retriever)

questions = ["What is SiloContainer?",
             "How many silo containers are there?"]

chat_history = []

for question in questions:
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result['answer']))
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")