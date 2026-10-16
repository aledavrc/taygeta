# Taygeta

Scientific exam bank and **MCP** retrieval system with **Human-in-the-Loop**
review. Taygeta ingests PDF/image documents, extracts questions and figures,
transcribes mathematics to Markdown/TeX, classifies them curricularly and stores
them **without solutions or correct answers**. It then allows searching real
exercises (RAG) and generating practice exams compiled to PDF with **Typst**.

> **Golden rule:** Taygeta transcribes and structures the visible content; it
> never solves the exercises, does not infer correct answers and does not store
> answer keys in any layer (schemas, staging, database, RAG, PDFs or MCP
> outputs).

---

## Table of contents

1. [Architecture](#architecture)
2. [Repository structure](#repository-structure)
3. [Requirements](#requirements)
4. [Installation and local configuration](#installation-and-local-configuration)
5. [Local usage](#local-usage)
6. [Deployment with Docker Compose](#deployment-with-docker-compose)
7. [OCR models](#ocr-models)
8. [Remote LLM](#remote-llm)
9. [SeaweedFS storage (S3)](#seaweedfs-storage-s3)
10. [HTTP endpoints and MCP tools](#http-endpoints-and-mcp-tools)
11. [Human-in-the-Loop workflow](#human-in-the-loop-workflow)
12. [Backups and restoration](#backups-and-restoration)
13. [Verification and testing](#verification-and-testing)
14. [Current limitations](#current-limitations)
15. [Security](#security)

---

## Architecture

**Hybrid Edge-to-Cloud** ingestion in two clearly separated passes:

```
 PDF/PNG/JPG
     │
     ▼
[1] Preparation  (ingest/image_extractor.py)
     PyMuPDF · 300 DPI rasterization · figure detection · BBox 0-1000
     padding 1.5% · temporary crops → upload to SeaweedFS
     │
     ▼
[2] Pass 1 · local OCR  (ingest/ocr_client.py)
     Local OpenAI-compatible service · transcribes to Markdown/TeX
     marks [ILEGIBLE] · does NOT classify or solve
     │
     ▼
[3] Single-pass structuring  (agents/structurer.py)
     Remote LLM + instructor + Pydantic v2 → EnrichedQuestion
     (formatted TeX + course/topic/subtopic + difficulty + prerequisites)
     │
     ▼
[4] Two-layer validation  (ingest/validator.py)
     Layer 1 local heuristics  +  Layer 2 remote semantics
     │
     ▼
[5] Staging (VALID / NEEDS_ATTENTION)  →  Human review (NiceGUI UI / MCP chat)
     │
     ▼
[6] Explicit confirmation → SQLite (taygeta.db, FTS5 search)
     │
     ▼
[7] RAG + exam generation (Typst)  →  PDF / source ZIP
```

Cross-cutting components:

- **Canonical schemas** (`core/schema.py`): `QuestionRaw`, `EnrichedQuestion`,
  `Choice`, `FigureReference`, `TranscriptionIntermediate`.
- **Transpiler** (`core/transpiler.py`): Pandoc AST → Typst; TeX formulas are
  delegated to **tylax** (LaTeX↔Typst converter in Rust). LaTeX export and
  validation compilation with the Typst CLI.
- **Persistence** (`server/db.py`): SQLite + **FTS5** virtual table synchronized
  by triggers; taxonomy and staging.
- **Objects** (`server/object_storage.py`): boto3 client for SeaweedFS (S3).
- **UI** (`ui/nice/`): Human-in-the-Loop review in **NiceGUI** (FastAPI).
- **Server** (`server/mcp_server.py`): FastMCP tools, HTTP downloads and TTL
  cleaner.

---

## Repository structure

```
taygeta/
├── Dockerfile                     # main multi-stage image (uv + Pandoc + Typst)
├── .dockerignore
├── docker-compose.yml             # GPU-neutral base
├── docker-compose.rocm.yml        # AMD/ROCm override
├── docker-compose.cuda.yml        # NVIDIA/CUDA override
├── docker/
│   ├── ocr/
│   │   ├── Dockerfile.cpu
│   │   ├── Dockerfile.rocm
│   │   ├── Dockerfile.cuda
│   │   └── entrypoint.sh
│   └── seaweedfs/
│       └── s3.json.example        # example S3 credentials (no secrets)
├── pyproject.toml
├── uv.lock
├── .env.example
└── src/taygeta/
    ├── core/     schema.py · settings.py · urls.py · transpiler.py
    ├── agents/   structurer.py
    ├── ingest/   image_extractor.py · ocr_client.py · validator.py · pipeline.py
    ├── server/   db.py · object_storage.py · exam_generator.py · mcp_server.py
    └── ui/       nice/ (app.py · question_panel.py · figures.py · state.py) · shared.py · figure_editing.py
```

---

## Requirements

- **Python 3.11+** and [uv](https://docs.astral.sh/uv/).
- For the full compilation and OCR flow: **Pandoc** and **Typst** CLI
  (already installed and verified in the Docker image).
- **tylax** TeX→Typst converter: it is installed automatically as a Python
  dependency (native Rust wheel, `abi3`).
- **Docker** and **Docker Compose v2** for container deployment.
- Optional GPU: AMD/ROCm or NVIDIA/CUDA to accelerate local OCR.

---

## Installation and local configuration

```bash
# 1) Dependencies (creates/updates uv.lock consistently)
uv sync

# 2) Configuration: copy the example and fill in the values
cp .env.example .env
#   Edit .env (LLM keys, OCR endpoint, S3 credentials). NEVER commit .env.
```

Check that the environment is set up correctly:

```bash
uv run python -c "from taygeta.core.settings import get_settings; s=get_settings(); print(s.safe_summary())"
```

> `uv.lock` is versioned. After changing `pyproject.toml`, run `uv lock` or
> `uv sync` and confirm that `uv.lock` is updated consistently.

Quality tools:

```bash
uv run ruff check src
uv run ruff format --check src
```

---

## Local usage

### CLI ingestion (`taygeta-ingest`)

```bash
uv run taygeta-ingest /path/to/exam.pdf
# with corrective context:
uv run taygeta-ingest /path/to/exam.pdf --hint "The illegible condition is g(x)=x+1"
```

It prints a JSON summary of the staging (number of questions, status and
session).

### MCP + HTTP server (`taygeta-server`)

```bash
uv run taygeta-server          # FastMCP with streamable-http transport
# by default it listens on http://0.0.0.0:8000  (MCP_HOST / MCP_PORT)
```

It exposes the MCP protocol at `/mcp` and the HTTP routes `/health`,
`/download/{filename}` and `/assets/{asset_id}`.

### Human-in-the-Loop review interface (NiceGUI)

```bash
uv run taygeta-ui            # or: uv run python -m taygeta.ui.nice
# http://localhost:8501
```

The UI allows: uploading a PDF/PNG/JPG, processing it with the two-pass pipeline
in the background (without blocking the interface), reviewing each question (a
Markdown/TeX editor with preview, taxonomy with course/topic comboboxes,
advanced options with in-place re-analysis) and managing the figures on the
right («Figures» / «Original page» views, crop by dragging, add and delete)
before explicitly confirming persistence into SQLite.

Host and port are adjusted with `TAYGETA_UI_HOST` / `TAYGETA_UI_PORT`.

---

## Deployment with Docker Compose

Copy the environment example before bringing up the stack:

```bash
cp .env.example .env      # fill in LLM_*, OCR_MODEL_PATH, S3 credentials
```

Dual configuration (local/production):

```dotenv
APP_ENV=local                       # 'local' or 'production'
DATABASE_URL=                       # optional; if empty, DB_PATH is used
                                    # e.g. sqlite:////app/data/taygeta.db
```

- `APP_ENV` sets the runtime environment (`Settings.is_production`).
- `DATABASE_URL` allows pointing to another SQLite path without touching
  `DB_PATH`; non-SQLite URLs are not supported by the current layer and produce
  an explicit error.

### Base / CPU (GPU-neutral)

```bash
docker compose up --build
```

Services: `mcp-server` (8000), `ui` (8501), `ocr` (CPU, internal network only),
SeaweedFS (`seaweedfs-master`, `seaweedfs-volume`, `seaweedfs-filer`,
`seaweedfs-s3:8333`).

### Local with AMD/ROCm

```bash
docker compose -f docker-compose.yml -f docker-compose.rocm.yml up --build
```

### Production with NVIDIA/CUDA

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up --build
```

> **Do not mix** the ROCm and CUDA overrides in the same run.
> The base file is GPU-neutral: the devices, the NVIDIA runtime and
> `HSA_OVERRIDE_GFX_VERSION` only appear in the overrides.

Validate the configuration without bringing up the services:

```bash
docker compose config -q
docker compose -f docker-compose.yml -f docker-compose.rocm.yml config -q
docker compose -f docker-compose.yml -f docker-compose.cuda.yml config -q
```

Host requirements for GPU:

- **ROCm**: AMD driver + ROCm compatible with the base image; access to
  `/dev/kfd` and `/dev/dri`; `video`/`render` groups depending on the host.
  Adjust `AMDGPU_TARGETS` (e.g. `gfx1100` for RX 7900) before building the OCR
  image. Only if ROCm does not detect the GPU natively, define
  `HSA_OVERRIDE_GFX_VERSION` (e.g. `10.3.0` for RDNA2); **never leave it
  empty**: an empty value disables detection and OCR falls back to CPU. With
  `OCR_REQUIRE_GPU=1` (already enabled in the override) the container **aborts
  with a clear message** if it does not see the GPU, instead of starting
  degraded.
- **CUDA**: NVIDIA driver + NVIDIA Container Toolkit. Verify first:
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```
  To reserve specific GPUs, edit `docker-compose.cuda.yml` and change
  `count: all` to `count: 1` or `device_ids: ["0"]`.

---

## OCR models

The `ocr` service uses `llama-server` (llama.cpp) with an **OpenAI-compatible**
API and multimodal support. **The weights are not included in the repository or
in the image**: they live in the persistent `ocr-models` volume.

**Automatic download from Hugging Face**: if the weights are not present, the
entrypoint downloads them on startup using `OCR_MODEL_URL` and `OCR_MMPROJ_URL`.
By default **Qwen2-VL-7B-Instruct (GGUF, Q4_K_M)** is used plus its **`mmproj`**:

```dotenv
OCR_MODEL=Qwen2-VL-7B-Instruct-Q4_K_M
OCR_MODEL_PATH=/models/Qwen2-VL-7B-Instruct-Q4_K_M.gguf
OCR_MODEL_URL=https://huggingface.co/bartowski/Qwen2-VL-7B-Instruct-GGUF/resolve/main/Qwen2-VL-7B-Instruct-Q4_K_M.gguf
OCR_MMPROJ_PATH=/models/mmproj-Qwen2-VL-7B-Instruct-f16.gguf
OCR_MMPROJ_URL=https://huggingface.co/bartowski/Qwen2-VL-7B-Instruct-GGUF/resolve/main/mmproj-Qwen2-VL-7B-Instruct-f16.gguf
OCR_CONTEXT_SIZE=8192
OCR_CONCURRENCY=1
```

- Downloads go to a `.part` file and are renamed on completion (resumable):
  an interruption does not leave a half-finished file.
- If the file **already exists** in the volume, it is not downloaded again.
- You can also **mount the weights manually** in `ocr-models` (e.g. with
  `huggingface-cli`) and adjust the paths/URLs in `.env`.
- If there are no weights and no URLs are defined, the entrypoint fails with a
  clear message instead of being marked healthy.
- The first startup downloads several GB (model + projector); the volume name
  may vary (`<project>_ocr-models`), check it with `docker volume ls`.

**Verify that the OCR sees images and transcribes a formula:**

```bash
# 1) Service health
docker compose exec ocr curl -fsS http://localhost:8080/health

# 2) Health + model from the application client
docker compose exec mcp-server python -c \
  "from taygeta.ingest.ocr_client import build_ocr_client; c=build_ocr_client(); print(c.health(), c.model)"

# 3) Real transcription of a test image
docker compose exec -T mcp-server python - <<'PY'
import io
from PIL import Image, ImageDraw
from taygeta.ingest.ocr_client import build_ocr_client

img = Image.new("RGB", (640, 140), "white")
ImageDraw.Draw(img).text((20, 50), "Equation: x^2 + 2x + 1 = 0", fill="black")
buf = io.BytesIO(); img.save(buf, format="PNG")
result = build_ocr_client().transcribe_page(buf.getvalue(), page_index=0)
print(result.text)
PY
```

The response must be Markdown/TeX with the formula transcribed (and `[ILEGIBLE]`
if something could not be read).

---

## Remote LLM

Single-pass structuring (which also infers ``course``, ``topic``, ``subtopic``,
``difficulty`` and ``prerequisites``) and semantic validation (Layer 2) use a
**remote** LLM compatible with the OpenAI API, configured **independently** of
the OCR:

```dotenv
LLM_API_KEY=            # ONLY the credential (no endpoint or model)
LLM_BASE_URL=           # e.g. https://my-provider.example/v1
LLM_MODEL=              # identifier of the model chosen by the administrator
LLM_TIMEOUT_SECONDS=120
LLM_MAX_RETRIES=2
LLM_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2   # local embeddings
DUPLICATE_SIMILARITY_THRESHOLD=0.90          # cosine threshold (0-1) for a duplicate candidate
```

Diagnostics:

```bash
uv run python -c "from taygeta.core.settings import get_settings; s=get_settings(); print('configured:', s.llm_configured); s.require_llm()"

# Check the endpoint (list of models) without exposing the key in logs:
curl -s "${LLM_BASE_URL%/}/models" -H "Authorization: Bearer ${LLM_API_KEY}" | head
```

> No provider or model is fixed in the code. `LLM_API_KEY` is only the
> credential; it does not contain the endpoint, the model or parameters.

### Semantic deduplication

In addition to exact-hash deduplication, ingestion computes a **local**
**embedding** of the statement and its choices with `sentence-transformers`
(`LLM_EMBEDDING_MODEL`, no external API or `LLM_API_KEY`) and compares it by
cosine similarity against the already stored questions of the same course. If
it exceeds `DUPLICATE_SIMILARITY_THRESHOLD`, the question is flagged as a
candidate and an LLM judge decides whether it is mathematically the same
(ignoring notation/OCR variations). If confirmed, the new text is discarded and
the `source_reference` and prerequisites are merged into the existing question.
Solutions and correct answers are never stored.

The model weights are downloaded/cached the first time embeddings are generated
(just like the OCR ones); a local path can also be used in
`LLM_EMBEDDING_MODEL`.

To migrate already existing questions (generate embeddings and merge duplicates):

```bash
uv run python scripts/deduplicate_questions.py --db /path/to/taygeta.db --dry-run
uv run python scripts/deduplicate_questions.py --db /path/to/taygeta.db
```

If the model is not available (no weights or network), semantic deduplication is
disabled in a controlled way and exact-hash deduplication is retained.

---

## SeaweedFS storage (S3)

Definitive figures are stored as S3 objects with a **stable relative path**
(`/buckets/{bucket}/{key}`, e.g. `/buckets/exams/img.png`), never as local paths
or persistent signed URLs. The relative path is portable across environments and
is resolved to a full URL only when sending the information to the frontend or
to MCP.

```dotenv
S3_ENDPOINT_URL=http://seaweedfs-s3:8333   # internal endpoint (Docker network)
S3_PUBLIC_ENDPOINT=http://localhost:8333   # public endpoint for the browser (UI)
S3_PUBLIC_BASE_URL=                        # public base for /buckets/{bucket}/{key}
S3_BUCKET=taygeta-assets
S3_ACCESS_KEY_ID=change-me
S3_SECRET_ACCESS_KEY=change-me
S3_REGION=us-east-1
ASSET_PUBLIC_BASE_URL=                     # optional public/proxy base (legacy, UI)
ASSET_URL_TTL_SECONDS=300
```

- **Single-source credentials**: the S3 gateway generates its IAM configuration
  from `S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY` in `.env`, so the application
  and gateway credentials always match. The IAM file schema is documented in
  `docker/seaweedfs/s3.json.example`. Change them before any non-local
  deployment.
- **Relative paths → public URLs**: the models store
  `/buckets/{bucket}/{key}`. To send the information to the frontend or the MCP
  client, the full URL is composed by concatenating `S3_PUBLIC_BASE_URL`
  (`build_public_url` in `core/urls.py`); in the UI you can also use
  `ASSET_PUBLIC_BASE_URL`, the application's `/assets/{asset_id}` endpoint or
  `S3_PUBLIC_ENDPOINT` (internal host rewriting). The host of a signed URL is not
  rewritten after generating it (it would invalidate the signature).

**Verify the bucket (create/read/delete an object):**

```bash
docker compose exec mcp-server python - <<'PY'
from taygeta.server.object_storage import ObjectStorage
s = ObjectStorage()
s.ensure_bucket()
uri = s.store_asset_bytes(b"test", prefix="checks", extension=".txt")
print("uploaded:", uri, "exists:", bool(uri))
key = uri.split("/", 3)[3]
print("read:", s.get_bytes(key))
s.delete(key); print("deleted:", not s.exists(key))
PY
```

---

## HTTP endpoints and MCP tools

HTTP routes on port 8000 (`mcp-server`):

| Route                        | Description                                              |
|------------------------------|----------------------------------------------------------|
| `GET /health`                | Status for healthchecks.                                 |
| `POST /mcp`                  | MCP transport (streamable-http).                         |
| `GET /download/{filename}`   | Serves temporary PDFs/ZIPs from `/app/downloads` (validates). |
| `GET /assets/{asset_id}`     | Returns a figure from SeaweedFS (validated id).          |

FastMCP tools:

| Tool | Description |
|---|---|
| `stage_exam_upload(file_path)` | Processes a file from `uploads/` and creates a staging session. |
| `refine_staged_question(session_id, q_index, user_hint, manual_overrides)` | Re-analyzes with corrective context; preserves overrides. |
| `confirm_staged_upload(session_id)` | Persists the approved questions (idempotent). |
| `search_questions(course, topic, difficulty, limit)` | RAG retrieval by course/topic/difficulty. |
| `list_taxonomy()` | Catalog of courses, topics, subtopics and prerequisites. |
| `generate_practice_exam(course, topics, difficulty, num_questions)` | Generates and compiles a PDF exam (Typst). |
| `get_exam_source(exam_id, format)` | ZIP with `main.typ`/`main.tex`, `assets/` and `README.txt`. |

Test the download after generating an exam:

```bash
EXAM_ID=$(docker compose exec -T mcp-server ... )   # returns the exam_id
curl -fL "http://localhost:8000/download/exams/${EXAM_ID}.pdf" -o exam.pdf
curl -fL "http://localhost:8000/download/exam_source_${EXAM_ID}.zip" -o source.zip
```

---

## Human-in-the-Loop workflow

1. **Upload** a PDF/PNG/JPG (UI or `stage_exam_upload`). It is saved in
   `uploads/`.
2. The pipeline runs the two passes and the two-layer validation; each question
   ends up in staging as `VALID` or `NEEDS_ATTENTION` (with a technical reason).
3. **Review** in the UI: the left column shows the Markdown statement with its
   rendered preview, the type/choices and the metadata; the right column is
   dedicated to the figures with two tabs, **Figures** and **Original page**
   (the source document page, useful for detecting omitted figures). Each figure
   can be **deleted** (its `{{fig:N}}` is cleaned from the statement) or
   **manually cropped** by drawing a bounding box on the original page; the crop
   is overwritten in SeaweedFS and the preview is refreshed instantly. If the
   model omitted a real figure (false negative), **➕ Add new figure** loads the
   full page, lets you draw the box over the missing graphic and saves it as the
   next free `fig:N`, inserting `{{fig:N}}` at the end of the statement so you
   can move it to its place.
4. **Edit** statement, type, choices, difficulty (1–5) and prerequisites.
   Course and Topic are comboboxes: select a slug from the catalog or use
   **✏️ Write manually...** to create a new one. Provide corrective context and
   press **Re-analyze** if needed (all in the left panel, with **Advanced
   Options** for the context and the new prerequisites).
5. **Confirm** to persist into `taygeta.db`. Manual correction prevails over
   automatic transcription.

> No question is saved into the definitive database without explicit
> confirmation. An OCR or validation failure is **never** hidden by marking the
> question as valid.

---

## Backups and restoration

### SQLite (`taygeta-data`)

Consistent copy with the SQLite backup API (avoids copying with in-flight
writes):

```bash
docker compose exec -T mcp-server python - <<'PY'
import sqlite3
src = sqlite3.connect("/app/data/taygeta.db")
dst = sqlite3.connect("/app/data/taygeta-backup.db")
with dst:
    src.backup(dst)
src.close(); dst.close()
print("backup created at /app/data/taygeta-backup.db")
PY
docker compose cp mcp-server:/app/data/taygeta-backup.db ./taygeta-backup.db
```

Restore (with the stack stopped to avoid writes):

```bash
docker compose stop mcp-server ui
docker compose cp ./taygeta-backup.db mcp-server:/app/data/taygeta.db
docker compose start mcp-server ui
```

Alternative via volume (cold snapshot):

```bash
docker compose stop mcp-server ui
docker run --rm -v "$(basename "$PWD")_taygeta-data":/data -v "$PWD":/backup alpine \
  tar czf /backup/taygeta-data.tgz -C /data .
docker compose start mcp-server ui
```

### SeaweedFS

Back up the persistent volumes of master, volume and filer (cold):

```bash
docker compose stop seaweedfs-s3 seaweedfs-filer seaweedfs-volume seaweedfs-master
for v in seaweedfs-master-data seaweedfs-volume-data seaweedfs-filer-data; do
  docker run --rm -v "$(basename "$PWD")_${v}":/data -v "$PWD":/backup alpine \
    tar czf "/backup/${v}.tgz" -C /data .
done
docker compose start seaweedfs-master seaweedfs-volume seaweedfs-filer seaweedfs-s3
```

Restore by extracting each `.tgz` into its corresponding volume (with the
services stopped). The bucket and credentials are reconfigured via
`docker/seaweedfs/s3.json`.

---

## Verification and testing

Static checks:

```bash
uv run ruff check src
uv run ruff format --check src
```

Integration verifications performed during development (against real
implementations of SQLite, PyMuPDF, Pandoc and Typst; with doubles only for the
external services that were not deployed):

- **Schemas**: Pydantic v2 validation, slugs `^[a-z0-9_]+$`, difficulty 1–5,
  placeholder/figure correspondence and absence of solution fields.
- **SQLite/FTS5**: idempotent initialization, synchronization of `questions_fts`
  on inserts/updates/deletes and parameterized queries.
- **Extraction**: 300 DPI rasterization, denormalization with 1.5 % padding,
  figure cropping, discarding candidates that cover the full page (scans,
  backgrounds or frame drawings) and manual re-cropping from the UI.
- **Transpiler**: TeX→Typst conversion delegated to **tylax** (golden corpus of
  formulas: fractions, roots, `pmatrix` matrices, `cases`, `aligned`, scripts,
  summations/integrals/limits) and **real PDF compilation** with Typst.
- **Pipeline/validation**: strict `VALID`/`NEEDS_ATTENTION` classification and
  absence of persisted correct answers.
- **MCP/exams**: PDF generation, source ZIP (Typst and LaTeX) and download
  routes; `docker compose config` for base + ROCm + CUDA.

> The automated `tests/` suite (pytest) is beginning to incorporate these cases
> (delimiter sanitization, TeX→Typst conversion with tylax and figure editing).
> The cases that depend on a GPU or on a real LLM/OCR provider can only be
> completed on the deployment host.

---

## Current limitations

- **OCR transcription**: depends on the local vision model. Illegible text or
  symbols are marked as `[ILEGIBLE]` and require human review; content is not
  invented.
- **TeX → Typst conversion (tylax)**: covers a broad catalog of LaTeX symbols
  and macros (~700 symbols, `\newcommand`/`\def`, matrices, `cases`, fractions,
  roots, scripts, summations/integrals/limits). tylax is **non-strict**: an
  unknown command is kept as raw text (it does not throw) and is detected during
  the real compilation with Typst (`unknown variable: ...`); for unparseable
  LaTeX, tylax emits a `/* LaTeX Error ... */` marker that the transpiler
  raises to a descriptive error while keeping the original.
- **Mathematical spacing**: tylax separates atoms structurally, but very
  specific spacing cases may differ visually (e.g. `\text{...}` loses the
  padding spaces at the ends).
- **Alignment (`aligned`)**: tylax emits Typst's native alignment (`&`/`\`); it
  is advisable to visually review cases with more than one column.
- **External services**: semantic validation and structuring require a remote
  LLM; if it is not available, the questions remain in `NEEDS_ATTENTION` with
  the technical reason (they are not lost).
- **GPU/model verification**: real GPU detection and transcription with the
  downloaded model must be checked on the deployment host.

---

## Security

- **Isolation and persistence**: the database lives in `/app/data`, the
  documents in `/app/uploads` and temporary artifacts in `/app/downloads`;
  definitive figures live in SeaweedFS. None of this is versioned in Git.
- **S3**: the gateway uses configured credentials and does not allow anonymous
  public reading. Signed URLs are not persisted; they are generated on demand.
- **Paths**: MCP inputs (input files) and downloads validate that the path stays
  within the allowed directory (anti-traversal).
- **Processes**: `pandoc` and `typst` are invoked with list arguments, never
  with `shell=True`; the TeX→Typst conversion runs **in-process** with `tylax`
  (native Rust library, no extra subprocesses).
- **Secrets**: `LLM_API_KEY`, S3 credentials and tokens are never printed in
  logs; use `Settings.safe_summary()` (masks secrets) for diagnostics.
- **No solutions**: correct answers are not stored in the database, staging,
  RAG, generated PDFs or MCP outputs.
- **Cleanup**: the `/app/downloads` cleaner respects a 60-minute TTL and does
  not touch the database, original uploads or bucket objects.

---

## License

Proprietary use (`license = "Proprietary"` in `pyproject.toml`).
