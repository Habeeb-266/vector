!pip install pypdf
!pip install faiss-cpu

from pypdf import PdfReader

pdf_path = "/content/Python Quick Guide.pdf"
reader = PdfReader(pdf_path)

text = ""
for page in reader.pages:
    text += page.extract_text()

print(text[:500])  # show preview
from pypdf import PdfReader

pdf_path = "/content/Python Quick Guide.pdf"
reader = PdfReader(pdf_path)

text = ""
for page in reader.pages:
    text += page.extract_text()

print(text[:500])  # show preview

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_embeddings = model.encode(chunks)

index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
index.add(np.array(chunk_embeddings))
query = "what is python?"
query_embedding = model.encode([query])

D, I = index.search(np.array(query_embedding), 3)

print("Query:", query)
print("\nTop Answers:\n")

for idx in I[0]:
    print(chunks[idx], "\n")

