A Streamlit app that:

takes news/article URLs,

fetches and chunks the pages,

embeds them with OpenAI,

stores vectors in a local FAISS index,

lets you ask questions with cited sources.

Perfect for quick fintech research and Q&A over a small set of links.

✨ Features
Paste up to 3 URLs and build a local FAISS index

Ask natural-language questions about the pages

Answers with sources (links de-duplicated)

Deterministic responses (temperature = 0)

🧰 Tech Stack
Python, Streamlit

LangChain (RetrievalQA, text splitting)

OpenAI (embeddings + LLM)

FAISS (vector store)

📁 Project Structure
bash
Copy
Edit
.
├─ app.py
├─ .gitignore
├─ notebooks/               # optional, your scratch/nb files
└─ faiss_openai_index/      # created at runtime (saved FAISS index)
⚙️ Setup
1) Create/use a Python env that has FAISS
On macOS Apple Silicon, the easiest path is conda (you already have an env named faissok):

bash
Copy
Edit
conda activate faissok
# if you ever need to install:
# conda install -c conda-forge faiss-cpu -y
If you prefer pyenv/pip only, you’ll need Rosetta/x86_64 or to build FAISS from source. Conda is simpler.

2) Install dependencies
bash
Copy
Edit
pip install -U pip
pip install streamlit langchain>=0.2 langchain-openai langchain-community unstructured
# FAISS is already in the env (if not, see above).
3) Add your OpenAI key (don’t commit it)
Create .streamlit/secrets.toml:

toml
Copy
Edit
OPENAI_API_KEY = "sk-..."
Make sure your .gitignore includes:

bash
Copy
Edit
.env
*.env
.streamlit/secrets.toml
faiss_openai_index/
▶️ Run
bash
Copy
Edit
conda activate faissok
streamlit run app.py
Enter up to three URLs in the sidebar

Click Process URLs (fetch → split → embed → index)

Ask a question in the input box

See the answer and sources

🧩 How it works (quick tour)
Load: UnstructuredURLLoader pulls and parses each URL.

Split: RecursiveCharacterTextSplitter divides content into ~1k-char chunks (with overlap).

Embed: OpenAIEmbeddings produces embeddings for each chunk.

Index: FAISS.from_documents(...) builds the vector index.

Retrieve + Answer: RetrievalQAWithSourcesChain runs a grounded response over the top-K chunks and returns text + sources.

Persist: The index is saved to disk (faiss_openai_index/) using FAISS’s save_local/load_local (more robust than pickling).

🔧 Config you can tweak
chunk_size=1000, chunk_overlap=150 (splitting)

Retriever k=4 (number of chunks retrieved)

Model: gpt-4o-mini (in ChatOpenAI), temperature = 0, max_tokens = 500

🛡️ Security / Secrets
Never commit .env or secrets.toml.

If a secret was committed in the past, rotate the key and rewrite history (e.g., git filter-repo --invert-paths --path path/to/file --force) before pushing.

🧪 Minimal code (what’s in app.py)
Your app uses the modern LangChain imports:

python
Copy
Edit
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
And persists the index like this:

python
Copy
Edit
vs.save_local("faiss_openai_index")
vs = FAISS.load_local("faiss_openai_index", embeddings, allow_dangerous_deserialization=True)
❗ Troubleshooting
“No module named ‘faiss’”

You’re likely not in the env with FAISS. Run:

bash
Copy
Edit
which python
python -c "import sys, faiss; print(sys.executable); print(faiss.__version__)"
Expect to see /anaconda3/envs/faissok/bin/python and a version (e.g., 1.9.0).
If not, conda activate faissok and try again.

FAISS install pain on macOS ARM (Apple Silicon)

Prefer conda install -c conda-forge faiss-cpu.

Pip often tries to compile from source and fails (SWIG/header errors).

GitHub push blocked (secret scanning)

Remove the secret file from the repo, add to .gitignore, rotate the key, and rewrite history:

bash
Copy
Edit
git rm --cached notebooks/.env
echo -e ".env\n*.env\nnotebooks/.env" >> .gitignore
git commit -m "Remove .env and ignore"
git filter-repo --invert-paths --path notebooks/.env --force
git push -u origin main --force
Streamlit “ScriptRunContext” warning

Safe to ignore when running outside the full Streamlit app context (e.g., executing script code directly).

Always launch via streamlit run app.py.

✅ Roadmap (nice-to-haves)
URL auto-discovery via a news search step (Tavily, SerpAPI, etc.)

Cache embeddings per URL to skip re-indexing unchanged pages

CSV export of answers + citations

Basic UI polish (history panel, clear index button)





