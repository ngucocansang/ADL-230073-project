"""
RAG engine: indexing, retrieval and answer generation.

Pipeline
    documents (.pdf/.txt/.md) -> clean text -> structure-aware chunks
    -> nomic-embed-text vectors -> Chroma (cosine)          [indexing]
    question -> hybrid retrieval (vector + BM25, RRF fusion) -> top-K chunks
    -> grounded prompt (+ short chat history) -> Ollama LLM -> answer + sources
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from rag.bm25 import BM25
from rag.loaders import list_supported_files, load_documents, page_for_offset

NOT_FOUND_MESSAGE = "I could not find this information in the documents."


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@dataclass
class RAGSettings:
    data_dir: str = config.DATA_DIR
    vector_root: str = config.VECTOR_ROOT
    collection: str = config.COLLECTION
    domain: str = config.DOMAIN
    chunk_size: int = config.CHUNK_SIZE
    chunk_overlap: int = config.CHUNK_OVERLAP
    structure_aware_split: bool = True
    min_chunk_chars: int = 150               # tiny fragments are merged into the next chunk
    ollama_url: str = config.OLLAMA_URL
    embed_model: str = config.EMBED_MODEL
    embed_prefixes: bool = config.EMBED_PREFIXES
    distance: str = "cosine"                 # "cosine" or "l2" (baseline)
    llm_model: str = config.LLM_MODEL
    temperature: float = config.TEMPERATURE
    num_ctx: int = config.NUM_CTX
    max_tokens: int = config.MAX_TOKENS          # -1 = unlimited (sample-code behaviour)
    top_k: int = config.TOP_K
    retrieval_mode: str = config.RETRIEVAL_MODE   # hybrid | vector | bm25
    score_threshold: float = config.SCORE_THRESHOLD
    history_turns: int = config.HISTORY_TURNS
    prompt_style: str = "grounded"           # "grounded" (v2), "grounded_v1" or "baseline"

    def index_key(self) -> str:
        """Settings that change the vectors -> separate index folder."""
        key = f"{self.embed_model}|{self.embed_prefixes}|{self.distance}|" \
              f"{self.chunk_size}|{self.chunk_overlap}|{self.structure_aware_split}"
        if self.min_chunk_chars:          # keeps the old key for unmerged indexes
            key += f"|{self.min_chunk_chars}"
        return hashlib.md5(key.encode()).hexdigest()[:8]


def baseline_settings(**overrides) -> RAGSettings:
    """Exactly the behaviour of the course's sample code (run_txt.py)."""
    s = RAGSettings(chunk_size=1000, chunk_overlap=200, structure_aware_split=False,
                    embed_prefixes=False, distance="l2", llm_model="qwen2.5:0.5b",
                    temperature=0.7, top_k=5, retrieval_mode="vector",
                    score_threshold=0.3, history_turns=0, prompt_style="baseline",
                    max_tokens=-1, min_chunk_chars=0)
    return replace(s, **overrides)


# --------------------------------------------------------------------------- #
# Embeddings with nomic task prefixes
# --------------------------------------------------------------------------- #
class PrefixedEmbeddings(Embeddings):
    """nomic-embed-text expects 'search_document:' / 'search_query:' prefixes."""

    def __init__(self, base: Embeddings, enabled: bool = True):
        self.base, self.enabled = base, enabled

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.enabled:
            texts = ["search_document: " + t for t in texts]
        return self.base.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.base.embed_query(("search_query: " + text) if self.enabled else text)


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
BASELINE_PROMPT = """You are a helpful AI assistant that answers questions based on provided documents.

CONTEXT FROM DOCUMENTS:
{context}
---
QUESTION: {question}

INSTRUCTIONS:
1. If the context contains relevant information, answer based on that.
2. If the context is insufficient, use your general knowledge.
3. Be concise - 2-4 sentences unless more detail is needed.

ANSWER:"""

# v1 (first attempt, kept for the evaluation in the report)
GROUNDED_PROMPT_V1 = """You are a helpful assistant that answers questions about {domain}.
Below are numbered passages retrieved from the documents. Each passage shows its source file and page.

CONTEXT:
{context}
{history}
QUESTION: {question}

INSTRUCTIONS:
1. Answer ONLY with information stated in the context passages. Do not use outside knowledge and do not invent facts.
2. Copy course codes, course titles, credit numbers and other numbers exactly as written in the context.
3. If the question asks for a list (courses, requirements, outcomes, careers), give the complete list as bullet points.
4. If the passages do not contain the answer, reply exactly: "{not_found}"
5. End your answer with the page number(s) you used, for example: (p. 6).

ANSWER:"""

# v2 (final). v1 made larger models over-refuse ("reply exactly ...") and give
# terse, oddly formatted answers ("End with the page numbers"), see report Sec. 6.
GROUNDED_PROMPT = """You are a helpful assistant that answers questions about {domain}, using passages retrieved from the documents.

CONTEXT FROM DOCUMENTS:
{context}
---
{history}QUESTION: {question}

INSTRUCTIONS:
1. Answer the question using the information in the context, in complete sentences.
2. If the question asks for a list, include every item that appears in the context.
3. Copy course codes, names and numbers exactly as they appear in the context. Do not add facts that are not in the context.
4. Only if the context contains nothing related to the question, say: "{not_found}"

ANSWER:"""

_FOLLOWUP = re.compile(r"\b(it|its|this|that|these|those|they|them|their|he|she|the course|"
                       r"the same|what about|and the|how about)\b", re.I)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class RAGEngine:
    def __init__(self, settings: Optional[RAGSettings] = None, rebuild: bool = False,
                 verbose: bool = True):
        self.s = settings or RAGSettings()
        self.verbose = verbose
        base = OllamaEmbeddings(model=self.s.embed_model, base_url=self.s.ollama_url)
        self.embeddings = PrefixedEmbeddings(base, self.s.embed_prefixes)
        self._llms: Dict[Tuple[str, float, int], OllamaLLM] = {}
        self.index_info: Dict = {}
        self.db = self._load_or_build(rebuild)
        self._build_keyword_index()

    # ------------------------------------------------------------- logging
    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------- indexing
    @property
    def index_dir(self) -> Path:
        name = Path(self.s.data_dir).resolve().name
        return Path(self.s.vector_root) / f"{name}_{self.s.index_key()}"

    def _fingerprint(self) -> Dict[str, str]:
        """sha1 of every document -> detects added / changed / removed files."""
        fp = {}
        for p in list_supported_files(self.s.data_dir):
            fp[p.name] = hashlib.sha1(p.read_bytes()).hexdigest()
        return fp

    def _splitter(self) -> RecursiveCharacterTextSplitter:
        if self.s.structure_aware_split:
            # Prefer to cut right before a course heading ("CS311: ..."), so each
            # course description stays in one chunk; then paragraphs, lines, sentences.
            separators = [r"\n(?=[A-Z]{2,4} ?\d{3}[a-z]{0,2}(?:/\d{3})?: )",
                          r"\n\n", r"\n", r"\. ", r" ", r""]
            return RecursiveCharacterTextSplitter(
                chunk_size=self.s.chunk_size, chunk_overlap=self.s.chunk_overlap,
                separators=separators, is_separator_regex=True,
                keep_separator="start", add_start_index=True)
        return RecursiveCharacterTextSplitter(
            chunk_size=self.s.chunk_size, chunk_overlap=self.s.chunk_overlap,
            add_start_index=True)

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        chunks = self._splitter().split_documents(documents)
        # Merge tiny fragments (stray headings such as "Program Learning Outcomes
        # (PLOs)") into the following chunk of the same file: alone they carry no
        # answer but still match many queries and waste top-K slots.
        merged: List[Document] = []
        carry: Optional[Document] = None
        for c in chunks:
            text = c.page_content.strip()
            if carry is not None and carry.metadata["source"] == c.metadata["source"]:
                c = Document(page_content=carry.page_content + "\n" + text,
                             metadata=carry.metadata)
                text = c.page_content
            elif carry is not None:
                merged.append(carry)
            carry = None
            if len(text) < self.s.min_chunk_chars:
                carry = Document(page_content=text, metadata=c.metadata)
                continue
            merged.append(Document(page_content=text, metadata=c.metadata))
        if carry is not None and len(carry.page_content) >= 20:
            merged.append(carry)

        out = []
        for i, c in enumerate(merged):
            md = dict(c.metadata)
            offsets = md.pop("page_offsets", "")
            md["page"] = page_for_offset(offsets, md.get("start_index", 0))
            md["chunk_id"] = f"{md['source']}#{i}"
            out.append(Document(page_content=c.page_content, metadata=md))
        return out

    def _chroma(self) -> Chroma:
        kwargs = dict(collection_name=self.s.collection,
                      embedding_function=self.embeddings,
                      persist_directory=str(self.index_dir))
        if self.s.distance == "cosine":
            kwargs["collection_metadata"] = {"hnsw:space": "cosine"}
        return Chroma(**kwargs)

    def _load_or_build(self, rebuild: bool) -> Chroma:
        data_dir = Path(self.s.data_dir)
        if not data_dir.exists() or not list_supported_files(data_dir):
            raise FileNotFoundError(f"No .pdf/.txt/.md files found in '{data_dir}'")
        manifest_path = self.index_dir / "manifest.json"
        fingerprint = self._fingerprint()
        if manifest_path.exists() and not rebuild:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("files") == fingerprint:
                self._log(f"Loading existing vector store: {self.index_dir}")
                self.index_info = manifest
                return self._chroma()
            self._log("Documents changed since last indexing -> rebuilding index.")
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self._log(f"Indexing documents in '{data_dir}' ...")
        t0 = time.time()
        documents = load_documents(data_dir, verbose=self.verbose)
        chunks = self.chunk_documents(documents)
        self._log(f"  {len(documents)} file(s) -> {len(chunks)} chunks "
                  f"(size={self.s.chunk_size}, overlap={self.s.chunk_overlap})")
        db = self._chroma()
        batch = 64
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            db.add_documents(part, ids=[c.metadata["chunk_id"] for c in part])
        self.index_info = {
            "files": fingerprint, "num_chunks": len(chunks),
            "settings": asdict(self.s), "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seconds": round(time.time() - t0, 1),
        }
        manifest_path.write_text(json.dumps(self.index_info, indent=2))
        self._log(f"  Indexed in {self.index_info['seconds']} s -> {self.index_dir}")
        return db

    def _build_keyword_index(self):
        data = self.db.get(include=["documents", "metadatas"])
        self.all_chunks = [Document(page_content=t, metadata=m)
                           for t, m in zip(data["documents"], data["metadatas"])]
        self.bm25 = BM25([d.page_content for d in self.all_chunks])
        self.index_info.setdefault("num_chunks", len(self.all_chunks))

    # ------------------------------------------------------------- retrieval
    def _similarity(self, distance: float) -> float:
        if self.s.distance == "cosine":
            return 1.0 - distance
        return 1.0 / (1.0 + distance)        # same conversion as the sample code

    def retrieval_query(self, question: str, history: Optional[List[Dict]] = None) -> str:
        """Make follow-up questions ('what are its prerequisites?') self-contained
        by prepending the previous user question."""
        if not history or self.s.history_turns <= 0:
            return question
        prev_user = [h["content"] for h in history if h["role"] == "user"]
        if prev_user and (len(question.split()) <= 6 or _FOLLOWUP.search(question)):
            return prev_user[-1] + " " + question
        return question

    def retrieve(self, query: str, top_k: Optional[int] = None) -> Tuple[List[Dict], float]:
        """Return (top-k sources, best vector similarity)."""
        k = top_k or self.s.top_k
        n = len(self.all_chunks)
        vec = self.db.similarity_search_with_score(query, k=min(n, max(k, 40)))
        vec_ranked = [(d.metadata.get("chunk_id"), d, self._similarity(dist)) for d, dist in vec]
        sim_by_id = {cid: sim for cid, _, sim in vec_ranked}
        best_sim = max(sim_by_id.values()) if sim_by_id else 0.0

        if self.s.retrieval_mode == "vector":
            ranked = [(cid, d, sim) for cid, d, sim in vec_ranked]
        else:
            bm = self.bm25.search(query, k=40)
            bm_ranked = [(self.all_chunks[i].metadata.get("chunk_id"), self.all_chunks[i], s)
                         for i, s in bm]
            if self.s.retrieval_mode == "bm25":
                ranked = bm_ranked
            else:  # hybrid: Reciprocal Rank Fusion
                fused: Dict[str, float] = {}
                docs: Dict[str, Document] = {}
                for lst in (vec_ranked[:40], bm_ranked):
                    for rank, (cid, d, _) in enumerate(lst):
                        fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank + 1)
                        docs[cid] = d
                order = sorted(fused, key=fused.get, reverse=True)
                ranked = [(cid, docs[cid], fused[cid]) for cid in order]

        sources = []
        for rank, (cid, d, _) in enumerate(ranked[:k], start=1):
            sources.append({
                "rank": rank,
                "chunk_id": cid,
                "source": str(d.metadata.get("source", "Unknown")),
                "page": int(d.metadata.get("page", 0) or 0),
                "score": round(float(sim_by_id.get(cid, 0.0)), 4),
                "content": d.page_content,
            })
        return sources, best_sim

    # ------------------------------------------------------------- prompting
    def build_prompt(self, question: str, sources: List[Dict], used_fallback: bool,
                     history: Optional[List[Dict]] = None) -> str:
        if self.s.prompt_style == "baseline":
            if used_fallback:
                context = "No sufficiently relevant documents found. Use your general knowledge."
            else:
                context = "\n\n---\n\n".join(f"[{s['source']}]\n{s['content']}" for s in sources)
            return BASELINE_PROMPT.replace("{context}", context).replace("{question}", question)

        if self.s.prompt_style == "grounded_v1":
            context = "\n\n".join(
                f"[{i}] ({s['source']}, p. {s['page']})\n{s['content']}" if s["page"]
                else f"[{i}] ({s['source']})\n{s['content']}"
                for i, s in enumerate(sources, start=1))
            template = GROUNDED_PROMPT_V1
        else:
            context = "\n\n---\n\n".join(
                f"[{s['source']}, page {s['page']}]\n{s['content']}" if s["page"]
                else f"[{s['source']}]\n{s['content']}" for s in sources)
            template = GROUNDED_PROMPT
        hist = ""
        if history and self.s.history_turns > 0:
            turns = history[-2 * self.s.history_turns:]
            lines = [f"{'User' if h['role'] == 'user' else 'Assistant'}: {h['content'][:600]}"
                     for h in turns]
            hist = "PREVIOUS CONVERSATION (use it only to understand follow-up questions):\n" + \
                   "\n".join(lines) + "\n\n"
        return (template.replace("{domain}", self.s.domain)
                .replace("{context}", context).replace("{history}", hist)
                .replace("{not_found}", NOT_FOUND_MESSAGE).replace("{question}", question))

    # ------------------------------------------------------------- generation
    def llm(self, model: Optional[str] = None, temperature: Optional[float] = None) -> OllamaLLM:
        model = model or self.s.llm_model
        temperature = self.s.temperature if temperature is None else temperature
        key = (model, temperature, self.s.num_ctx)
        if key not in self._llms:
            self._llms[key] = OllamaLLM(model=model, base_url=self.s.ollama_url,
                                        temperature=temperature, num_ctx=self.s.num_ctx,
                                        num_predict=self.s.max_tokens)
        return self._llms[key]

    def prepare(self, question: str, history: Optional[List[Dict]] = None) -> Dict:
        """Retrieval + prompt construction (everything except the LLM call)."""
        t0 = time.time()
        rq = self.retrieval_query(question, history)
        sources, best_sim = self.retrieve(rq)
        if self.s.prompt_style == "baseline":
            avg = sum(s["score"] for s in sources) / len(sources) if sources else 0
            used_fallback = not sources or avg < self.s.score_threshold
        else:
            used_fallback = not sources or best_sim < self.s.score_threshold
        prompt = self.build_prompt(question, sources, used_fallback, history)
        return {"question": question, "retrieval_query": rq, "sources": sources,
                "best_similarity": round(best_sim, 4), "used_fallback": used_fallback,
                "prompt": prompt, "retrieve_s": round(time.time() - t0, 2)}

    def ask(self, question: str, history: Optional[List[Dict]] = None,
            model: Optional[str] = None) -> Dict:
        prep = self.prepare(question, history)
        t0 = time.time()
        if prep["used_fallback"] and self.s.prompt_style != "baseline":
            answer = NOT_FOUND_MESSAGE      # nothing relevant -> do not let the model guess
        else:
            answer = self.llm(model).invoke(prep["prompt"]).strip()
        prep.update(answer=answer, model=model or self.s.llm_model,
                    generate_s=round(time.time() - t0, 2))
        return prep

    def stream(self, prep: Dict, model: Optional[str] = None,
               temperature: Optional[float] = None) -> Iterator[str]:
        """Token stream for the web UI."""
        if prep["used_fallback"] and self.s.prompt_style != "baseline":
            yield NOT_FOUND_MESSAGE
            return
        yield from self.llm(model, temperature).stream(prep["prompt"])


# --------------------------------------------------------------------------- #
# Ollama helpers
# --------------------------------------------------------------------------- #
def ollama_models(url: str = config.OLLAMA_URL) -> List[str]:
    """Names of the models available on the Ollama server ([] if unreachable)."""
    try:
        r = requests.get(url.rstrip("/") + "/api/tags", timeout=3)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))
    except Exception:
        return []
