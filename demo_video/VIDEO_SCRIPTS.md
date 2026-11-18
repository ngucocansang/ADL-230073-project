# Video demo scripts (Task 6)

Four short screen recordings are required, one per task (each worth 5%).
Target length: 2–4 minutes each. Record the terminal / browser full-screen
(macOS: `Cmd+Shift+5`; Windows: Xbox Game Bar `Win+Alt+R`; Linux: OBS or SimpleScreenRecorder).
Speak while you click; the rubric asks you to show **the implementation, its purpose and the output**.

Suggested file names (put them in this folder, or upload to Google Drive and paste the links
in `links.md` + the report + the Canvas comment if they exceed the upload limit):

| File | Task |
|---|---|
| `task1_pdf_support.mp4` | Task 1 – PDF file support |
| `task2_handbook_chatbot.mp4` | Task 2 – CS major handbook chatbot |
| `task3_docker.mp4` | Task 3 – Docker deployment |
| `task4_web_ui.mp4` | Task 4 – Web interface |

Before recording: `ollama serve` is running and the models are pulled
(`nomic-embed-text`, `qwen2.5:0.5b`, `qwen2.5:1.5b`, optionally `qwen2.5:3b`).
Delete `source_code/vector_store/` once so the indexing step is visible on camera.

---

## Video 1 – Task 1: PDF support (≈3 min)

1. **Purpose (20 s).** "The sample code only read `.txt` files. Task 1 extends the loader so the
   system can index PDF documents."
2. **Code walkthrough (60 s).** Open `source_code/rag/loaders.py`:
   - `load_pdf()` – pypdf extracts the selectable text of every page; the page start offsets are stored so
     each chunk can be mapped back to its page number.
   - `clean_text()` – fixes ligature artefacts produced by the handbook fonts
     (`Descrip;on → Description`, `A2ributes → Attributes`), removes table-of-contents dot leaders.
   - `load_documents()` – picks the loader by file extension (`.pdf`, `.txt`, `.md`).
   - Mention the warning for scanned PDFs (no selectable text → would need OCR).
3. **Run on a PDF that is not the handbook (60 s).** The project description PDF is in `data/pdf_demo/`:
   ```bash
   cd source_code
   python run_rag.py --data-dir data/pdf_demo --domain "the CS311 final project description" \
       --rebuild -q "How much is the Docker container deployment task worth?"
   ```
   Show: "Loading: Final-project-ADL.pdf", number of chunks indexed, the answer (20%), and the
   sources with **page numbers**.
4. **Mixed folder (optional, 20 s).** `python run_rag.py --data-dir data/txt_demo -q "Where is the FUV campus?"`
   shows the original `.txt` support still works.
5. **Wrap-up (10 s).** "PDF → text → chunks → embeddings → Chroma; answers cite page numbers."

## Video 2 – Task 2: CS major handbook chatbot (≈4 min)

1. **Purpose (20 s).** "I chose the B.S. Computer Science handbook (19 pages). The chatbot answers
   questions on requirements, courses, learning outcomes and policies."
2. **Pipeline (60 s).** Open `rag/engine.py` and point at: structure-aware chunking (cut before
   `CS311:` headings), nomic task prefixes + cosine distance, hybrid retrieval (vector + BM25 with
   Reciprocal Rank Fusion), the grounded prompt with the "I could not find…" rule.
3. **Live questions (90 s).** `python run_rag.py --chat` then ask:
   - `What are the prerequisites for CS311 Advanced Deep Learning?` → CS313 ✅
   - `What are its learning outcomes?` *(follow-up – history resolves "its")*
   - `What are the eligibility criteria for the Capstone?` → GPA 3.5 + application ✅
   - `What is the tuition fee?` → "I could not find this information" ✅ (no hallucination)
   - `What degree is awarded after completing the requirements?` → shows the handbook's own
     "Applied Mathematics" typo – an **ambiguous** case.
4. **Evaluation (60 s).** Open `eval/questions.json` (32 questions), then
   `eval/results/summary.md` and `eval/results/retrieval_ablation.md`. Show the table comparing
   baseline vs improved and 0.5b / 1.5b / 3b, and one example each of a correct, incomplete,
   incorrect and ambiguous answer (they are listed in the report, Section 5).
   Command (do not run the full thing on camera – it takes a while):
   `python evaluate.py --retrieval` (fast) and `python evaluate.py --generate improved-1.5b`.
5. **Wrap-up (10 s).**

## Video 3 – Task 3: Docker deployment (≈3 min)

1. **Purpose (20 s).** "Docker makes the system portable and reproducible: one command starts
   Ollama, downloads the models and starts the chatbot."
2. **Files (60 s).** Show `docker/Dockerfile` (python:3.11-slim, requirements layer cached, non-root
   user, health check, entrypoint), `docker/entrypoint.sh` (waits for Ollama, `web` / `cli` modes) and
   `docker/docker-compose.yml` (3 services: `ollama`, `model-init`, `app`; volumes for models,
   the handbook folder and the Chroma index).
3. **Build & run (60 s).**
   ```bash
   cd docker
   docker compose up --build -d
   docker compose ps          # ollama healthy, model-init exited (0), app healthy
   docker compose logs -f app # "Indexing documents..." then "You can now view your Streamlit app"
   ```
   (macOS alternative using the host's Ollama + Apple GPU:
   `docker compose -f docker-compose.host-ollama.yml up --build`)
4. **Test (30 s).** Open http://localhost:8501 and ask one question; also show the CLI inside the
   container: `docker compose exec app python run_rag.py -q "How many credits does the minor require?"`
5. **Volumes (20 s).** `docker compose down` then `up -d` again → log says *"Loading existing vector
   store"* (index persisted). `docker compose down -v` removes everything.

## Video 4 – Task 4: Web interface (≈3 min)

1. **Purpose (15 s).** "A Streamlit web UI so non-technical students can use the chatbot in a browser."
2. **Code (40 s).** `source_code/app.py`: cached engine, sidebar settings, `st.chat_message`
   history stored in `st.session_state`, token streaming with `st.write_stream`, sources expander.
3. **Demo (2 min).**
   - Click an example question button.
   - Type a follow-up: *"What are its prerequisites?"* – show that history works.
   - Open **Sources**: page numbers, similarity scores, retrieved text.
   - Change the model in the sidebar (0.5b → 3b) and ask the same question; compare answer and time.
   - Ask an out-of-scope question → "could not find" + info box.
   - Click **Export** (download the chat as Markdown) and **New chat** (clears history).
4. **Wrap-up (10 s).**
