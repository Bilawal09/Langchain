from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model = 'text-embedding-3-large',dimensions =300)
 
documents = [
  'Umar Kalyal is the Roll No "1" in BS_AI Batch(2022-26)',
  'Hassan Jahangir is the Roll No "10" in BS_AI Batch(2022-26)',
  'Ammar Ali is the Roll No "12" in BS_AI Batch(2022-26)',
  'Bilawal Khan is the Roll No "14" in BS_AI Batch(2022-26)',
  'Sameer  Nazakat is the Roll No "42" in BS_AI Batch(2022-26)'
]

query = 'tell me about Hassan Jahangir'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_documents(query)

scores = cosine_similarity(query_embedding, doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print('similarity score is:',score)