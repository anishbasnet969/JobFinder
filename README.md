# JobFinder

An intelligent job recommendation platform that uses AI-powered semantic search and reranking to match job seekers with the most relevant job opportunities.

## Technology Stack

- **Backend**: Python, FastAPI
- **Database**: PostgreSQL with pgvector extension
- **Caching**: Redis
- **Task Queue**: Celery with RabbitMQ
- **AI/ML**: Google Gemini (embeddings), Cohere (reranking), LangChain
- **Data Processing**: Pandas, PyMuPDF
- **Containerization**: Docker

---

## 📋 Prerequisites

- **Docker** installed on your system
- **API Keys**:
  - Google Gemini API key (for embeddings) - [Get it here](https://aistudio.google.com/app/apikey)
  - Cohere API key (for reranking) - [Get it here](https://dashboard.cohere.com/api-keys)

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/anishbasnet969/JobFinder.git
cd JobFinder
```

### 2. Environment Configuration

Copy the example environment file and configure it with your API keys:

```bash
cp .env.example .env
```

Edit the `.env` file and add your API keys:

```env
# Required: Add your API keys
GOOGLE_API_KEY=your_google_api_key_here
COHERE_API_KEY=your_cohere_api_key_here

# Optional: Customize these settings if needed
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
POSTGRES_USER=app
POSTGRES_PASSWORD=app
POSTGRES_DB=jobfinderdb
JOB_INGEST_ON_STARTUP=true
JOB_INGEST_LIMIT=500
```

**Important Environment Variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google Gemini API key for embeddings | **Required** |
| `COHERE_API_KEY` | Cohere API key for reranking | **Required** |
| `JOB_INGEST_ON_STARTUP` | Auto-load jobs from CSV on startup | `true` |
| `JOB_INGEST_LIMIT` | Number of jobs to load from dataset | `500` |
| `CACHE_RESUME_TTL` | Resume cache duration (seconds) | `86400` |
| `VECTOR_SEARCH_CANDIDATES` | Initial candidates for vector search | `100` |

### 3. Database Setup

The database is automatically set up when you run the application using Docker Compose. The setup includes:

- **PostgreSQL 18** with **pgvector** extension for vector similarity search
- **Automatic Migrations**: Alembic migrations run automatically on startup
- **Schema Creation**: All necessary tables and indexes are created

**Database Schema:**
- `jobs` table: Stores job descriptions with embeddings
- pgvector extension: Enables efficient similarity search
- Indexes: Optimized for vector search performance

### 4. Running the Application

#### Development Mode

Start all services (API, Database, Redis, RabbitMQ, Celery workers):

```bash
docker-compose -f docker-compose.dev.yml up --build
```

**Services will be available at (local):**
- **API (local):** http://localhost:5000
- **API Docs (local, Swagger):** http://localhost:5000/docs
- **RabbitMQ Management (local):** http://localhost:15672 (guest/guest)
- **Flower (local, Celery monitoring):** http://localhost:5555

#### Production Mode

```bash
docker-compose -f docker-compose.prod.yml up --build -d
```

#### Stopping the Application

Development (local):

```bash
docker-compose -f docker-compose.dev.yml down
# To also remove volumes (database data)
docker-compose -f docker-compose.dev.yml down -v
```

Production:

```bash
docker-compose -f docker-compose.prod.yml down
# If you started services in detached mode, you can stop them first
docker-compose -f docker-compose.prod.yml stop
# To also remove volumes (data) in production
docker-compose -f docker-compose.prod.yml down -v
```

---

## 🧪 Testing Instructions

### Testing with Sample Data

The repository includes sample data for testing. Navigate to http://localhost:5000/docs (local) to access the interactive API documentation (Swagger UI).

**Get Job Recommendations:**
1. Go to `POST /api/recommendations/for-resume`
2. Upload one of the sample PDFs from `samples/sample_resumes/pdfs/`
3. Receive ranked job recommendations with match scores and similarity details

**View Jobs:**
1. Go to `GET /api/jobs`
2. Browse available jobs with pagination
3. Get detailed job information by ID using `GET /api/jobs/{job_id}`

**Add a New Job:**
1. Go to `POST /api/jobs`
2. Provide job title and description
3. AI will automatically parse and structure the job posting

**View Performance Metrics:**
1. Go to `GET /api/metrics`
2. See CV parsing, recommendation generation, and database query performance
3. Use `GET /api/metrics/summary` for a quick overview

---

## 📦 Sample Data & Outputs

### Sample Resumes (6 PDFs)

Located in `samples/sample_resumes/pdfs/`:
- `BackendDeveloper.pdf`
- `DataScientist.pdf`
- `DevOps.pdf`
- `FrontendDeveloper.pdf`
- `FullStackDeveloper.pdf`
- `UI-Designer.pdf`

### Parsed Resume JSONs

Pre-parsed resumes following the **JSON Resume Schema** in `samples/sample_resumes/parsed_resumes_json/`:
- `BackendDeveloper.json`
- `DataScientist.json`
- `DevOps.json`
- `FrontendDeveloper.json`
- `FullStackDeveloper.json`
- `UI-Designer.json`

### Recommendation Results

Pre-generated recommendation results in `samples/sample_resumes/recommendations_json/`:
- `BackendDeveloperRecommendation.json`
- `DataScientistRecommendation.json`
- `DevOpsRecommendation.json`
- `FrontendDeveloperRecommendation.json`
- `FullStackDeveloperRecommendation.json`
- `UI-DesignerRecommendation.json`


### Sample Job Descriptions (12 Jobs)

Located in `samples/sample_job_descriptions/` (JSON and TXT formats):
- Back End Developer
- Civil Engineer
- Data Engineer
- Digital Marketing Specialist
- Front End Engineer
- Graphic Designer
- Human Resources Manager
- IT Manager
- Network Analyst
- Product Manager
- Social Worker
- Software Tester

### Job Dataset

Large job dataset in `datasets/Job/job_title_descriptions.csv`:
- **500+ job descriptions** (configurable with `JOB_INGEST_LIMIT`)
- Automatically loaded on startup when `JOB_INGEST_ON_STARTUP=true`
- Includes titles, descriptions, and company information

---

## 🛠️ Development

### Running Migrations

Migrations run automatically on startup. To create new migrations:

```bash
# Enter the app container
docker-compose -f docker-compose.dev.yml exec app bash

# Create a new migration
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head
```

### Monitoring

- **Celery Tasks (local)**: View task status at http://localhost:5555 (Flower)
- **RabbitMQ (local)**: Monitor queues at http://localhost:15672
- **API Metrics**: Available at `/api/metrics` endpoint (local when running locally)
- **Logs**: View with `docker-compose logs -f app` (run from project root)

---

## 📝 API Endpoints

### Recommendations

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/recommendations/for-resume` | Get job recommendations by uploading a resume PDF |

### Jobs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/jobs` | List all jobs (paginated) |
| `GET` | `/api/jobs/{job_id}` | Get specific job details by ID |
| `POST` | `/api/jobs` | Create a new job posting (auto-parsed) |
| `GET` | `/api/jobs/stats/count` | Get total number of jobs in database |

### Metrics

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/metrics` | Get all performance metrics with detailed stats |
| `GET` | `/api/metrics/summary` | Get simplified metrics summary |
| `GET` | `/api/metrics/timing/{metric_name}` | Get detailed timing stats for specific metric |
| `DELETE` | `/api/metrics` | Reset metrics (with optional pattern) |

See full API documentation (local) at http://localhost:5000/docs

---

## Acknowledgments

- Job dataset sourced from https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset