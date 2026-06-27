# RAG With MinIO

In this Repo, we will demonstrate how to use MinIO to build a Retrieval Augmented Generation(RAG) based chat application using commodity hardware.
* Use MinIO to store all the documents, processed chunks and the embeddings using the vector database.
* Use MinIO's bucket notification feature to trigger events when adding or removing documents to a bucket
* Webhook that consumes the event and process the documents using Langchain and saves the metadata and chunked documents to a metadata bucket
* Trigger MinIO bucket notification events for newly added or removed chunked documents
* A Webhook that consumes the events and generates embeddings and save it to the Vector Database (LanceDB) that is persisted in MinIO

## Architecture
![Architecture](media/rag-minio.jpg)

## Key Tools Used
* **MinIO**  - Object Store to persist all the Data
* **LanceDB** - Serverless open-source Vector Database that persists data in object store
* **Ollama** - To run LLM and embedding model locally (OpenAI API compatible)
* **FastAPI** - Server for the Webhooks that receives bucket notification from MinIO and exposes the Gradio App
* **FastAPI-users** - Ready-to-use and customizable users management for FastAPI
* **LangChain & Unstructured** - To Extract useful text from our documents and Chunk them for Embedding

### Models Uses
* **LLM** - Phi-3-128K (3.8B Parameters)
* **Embeddings** - Nomic Embed Text v1.5 ([Matryoshka Embeddings](https://arxiv.org/pdf/2205.13147)/ 768 Dim, 8K context)

## Setup
Install the required packages using the following command:
```bash
pip install -r requirements.txt
```

## Running the Application
You can follow the step by step described in the [Notebook](RAG-with-MinIO.ipynb) to run the application.

## Dependencies
```bash
pip install pandas 
pip install pylance
pip install "fastapi-users[sqlalchemy]" 
pip install "fastapi-users[oauth]"
pip install httpx-oauth
pip install aiosqlite
pip install boto3
pip install apscheduler
pip install lancedb
pip install langchain
pip install langchain-community
pip install langchain-text-splitters
pip install "unstructured[all-docs]"
pip install pytesseract
pip install uvicorn
```

![Chat Agent](media/chat-agent.png)

![Minio Buckets](media/minio-buckets.png)

![Minio Events](media/minio-events.png)

![Minio Event Binding](media/mino-event-binding.png)

## OCR Engine
We will use the Google’s Tesseract-OCR Engine [tesseract](https://tesseract-ocr.github.io/tessdoc/)

In mac we can install the engine like this:

``` 
$ brew install tesseract
```

In Linux we can install the engine like this:

```
sudo apt install tesseract-ocr
```

## Minio binaries

- [For Mac](https://dl.min.io/community/server/minio/release/darwin-arm64/archive/)
- [For Ubuntu](https://dl.min.io/community/server/minio/release/linux-amd64/archive/)
- [Check Minio Deployment](https://github.com/masalinas/doc-minio-docker)

## App running steps
Start minio locally
```bash
$ cd binaries
$ ./minio.RELEASE.2025-01-20T14-49-07Z server ./data
```

Start DockSphere Backend
```bash
$ python -m app.main
```

## App public urls

Minio Portal
```bash
http://localhost:9001
```

Chat UI
```bash
http://localhost:8808/chat
```

## Sample prompts
Sample prompts based on the documents uploaded called: 'Fusion_Models.pdf' and 'Thingtrack-Catalogo-WeldTracer.pdf'

```bash
Give me the authors from the paper called: Multimodal Human Activity Recognition using fusion strategies

AI: The authors of the paper titled "Multimodal Human Activity Recognition using fusion strategies" are Miguel 'A. Salinas, Antonio M. Lopez, Alejandro Castellanos, Juan C.' Alvarez, Diego 'Alvarez, Gonzalo Garc'ıa, David Mart'ınez, Ver'onica Cabanas, Ignacio Ara, Asier Ma˜nas, Raquel Bail'on, Vicente Mart'ın, and Ana Carbonell.

The document discusses the performance of various fusion strategies in multimodal human activity recognition (HAR) systems that rely on heterogeneous sensor data and feature representations. It highlights Random Forest-based fusion methods as favorable due to their balance between predictive performance, robustness, interpretability, and computational cost. These methods are particularly attractive for large cohort studies and digital health applications where scalability and reproducibility are essential.

The document also compares different late fusion strategies such as stacking Random Forest (stacking rf) and mixture of experts with Random Forest (moe rf), which consistently achieved the highest F1-scores across folds, indicating that ensemble-based decision-level integration is more effective than simple feature concatenation when combining heterogeneous sensor modalities.

In addition to these findings, the document presents experimental results showing that fusion models outperform individual classifiers in complex and real-world HAR scenarios by exploiting complementary strengths of multiple specialized models through techniques like Mixture of Experts (MoE) and stacking.
```

```bash
Some information about Weld Tracer from Thyngtrack company

AI: Weld Tracer is a robust and scalable real-time monitoring system for welding equipment, specifically designed by Edikal Technology. It monitors critical parameters of MIG/MAG welding processes such as tension, current, wire feed speed, and electrical consumption. The data collected can be stored in the cloud through CloudsWeldTracker, allowing users to download it for analysis on any device. This system offers external connection capabilities without requiring modifications to the equipment being monitored, ensuring no impact on warranties or guarantees of the machinery involved.

The document does not provide specific information about Thyngtrack company's Weld Tracer; therefore, I cannot answer questions related to it based on this text alone.
```

## Connection to your object storage server (minio)
First create an alias connection called **local** to your minio server

```shell
$ mc alias set local http://localhost:9000 minioadmin minioadmin
```

## App Open API
Open API Json Documentation
```shell
http://localhost:8808/openapi.json
```

Open API Swagger Documentation
```shell
http://localhost:8808/docs
```

Register a user in fastapi-users
```shell
curl -X 'POST' \
  'http://localhost:8808/api/v1/auth/register' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "email": "miguel@thingtrack.com",
  "password": "password",
  "name": "Miguel Salinas Gancedo",
  "is_active": true,
  "is_superuser": true,
  "is_verified": false
}'
{"id":"5eaa3d8f-ac73-4e05-b370-a6f2531792b4","email":"miguel@thingtrack.com","is_active":true,"is_superuser":false,"is_verified":false,"name":"Thingtrack User"}
```

Login session from credentials
```shell
curl -X 'POST' \
  'http://localhost:8808/api/v1/auth/jwt/login' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'username=masalinas.gancedo@gmail.com&password=!Thingtrack2010'

{"access_token":"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1ZWFhM2Q4Zi1hYzczLTRlMDUtYjM3MC1hNmYyNTMxNzkyYjQiLCJhdWQiOlsiZmFzdGFwaS11c2VyczphdXRoIl0sImV4cCI6MTc3NjAyOTc5MH0.Ub-WhJxoZiq6f3WiBdP88jBNBWZVVQhXAVyyBqSJQpk","token_type":"bearer"}
```

Get session from token
```shell
curl -X 'GET' \
  'http://localhost:8808/api/v1/auth/me' \
  -H 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1ZWFhM2Q4Zi1hYzczLTRlMDUtYjM3MC1hNmYyNTMxNzkyYjQiLCJhdWQiOlsiZmFzdGFwaS11c2VyczphdXRoIl0sImV4cCI6MTc3NjAyOTc5MH0.Ub-WhJxoZiq6f3WiBdP88jBNBWZVVQhXAVyyBqSJQpk'
{"id":"5eaa3d8f-ac73-4e05-b370-a6f2531792b4","email":"uo34525@uniovi.es","is_active":true,"is_superuser":false,"is_verified":true}
```

Logout session from token 
```shell 
curl -X 'POST' \
  'http://localhost:8808/api/v1/auth/jwt/logout' \
  -H 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1ZWFhM2Q4Zi1hYzczLTRlMDUtYjM3MC1hNmYyNTMxNzkyYjQiLCJhdWQiOlsiZmFzdGFwaS11c2VyczphdXRoIl0sImV4cCI6MTc3NjAyOTc5MH0.Ub-WhJxoZiq6f3WiBdP88jBNBWZVVQhXAVyyBqSJQpk' \
  -H 'Content-Type: application/json'
```

## Minio infraestructure configuration

1. Create **custom-corpus** bucket where save knowledge base
    ```shell
    $ mc mb local/custom-corpus
    ```

2. Create webhook event called **doc-webhook**
    ```shell
    $ mc admin config set local notify_webhook:doc-webhook \
    endpoint="http://localhost:8808/api/v1/document/notification" \
    queue_limit="1000"
    ```

3. Restart minio
    ```shell
    $ mc admin service restart local
    ```

4. Link the webhook event **doc-webhook** to the **custom-corpus** bucket
    ```shell
    $ mc event add local/custom-corpus arn:minio:sqs::doc-webhook:webhook \
    --event put,delete
    ```

5. Create **warehouse** bucket where save document tokens and lance-db database
    ```shell
    $ mc mb local/warehouse
    ```

6. Create webhook event called **metadata-webhook**
    ```shell
    $ mc admin config set local notify_webhook:metadata-webhook \
    endpoint="http://localhost:8808/api/v1/metadata/notification" \
    queue_limit="1000"
    ``` 

7. Restart minio
    ```shell
    $ mc admin service restart local
    ```

8. Link the webhook event **metadata-webhook** to the **warehouse** bucket
    ```shell
    $ mc event add local/warehouse arn:minio:sqs::metadata-webhook:webhook \
    --event put,delete \
    --prefix "metadata/" \
    --suffix ".json"
    ```

## Private RAG vs Public LLM (ChatGPT)

| Aspect                     | Private LLM + RAG                                                                   | Public LLM (ChatGPT with uploaded documents)              |
| -------------------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------- |
| **Data privacy**           | Full control. Documents stay inside your infrastructure.                            | Data is sent to a third-party provider.                   |
| **Security / compliance**  | Easier to meet strict regulations (GDPR, HIPAA, internal policies).                 | Depends on provider policies and enterprise agreements.   |
| **Data ownership**         | Complete ownership of documents and embeddings.                                     | Provider processes the data.                              |
| **Customization**          | Full control over prompts, chunking, embeddings, vector DB, and retrieval strategy. | Limited customization of retrieval or indexing.           |
| **Model choice**           | You can use open models (Llama, Mistral, etc.) or proprietary ones.                 | Limited to the models provided by the service.            |
| **Integration**            | Deep integration with internal systems (databases, APIs, knowledge bases).          | Integration typically via APIs with some limits.          |
| **Latency control**        | Can be optimized locally or near your infrastructure.                               | Depends on external service latency.                      |
| **Scalability**            | You must manage scaling and infrastructure.                                         | Automatically handled by the provider.                    |
| **Operational complexity** | High: requires infrastructure, monitoring, updates, GPUs, etc.                      | Very low: the provider manages everything.                |
| **Cost model**             | Fixed infrastructure cost (servers, GPUs, storage).                                 | Pay per token / request.                                  |
| **Maintenance**            | Must maintain models, vector DB, pipelines.                                         | No maintenance required.                                  |
| **Performance tuning**     | Full control of embeddings, retrieval strategies, and RAG architecture.             | Limited tuning possibilities.                             |
| **Reliability**            | Depends on your infrastructure reliability.                                         | Usually high availability from provider.                  |
| **Innovation speed**       | Slower updates unless you maintain them.                                            | Providers frequently release new models and improvements. |

## Some links

- [Minio](https://www.min.io/): MinIO AIStor delivers unmatched enterprise performance, scale, agility and economics for AI data, agentic computing, and analytics.

- [LangChain](https://www.langchain.com/): Observe, evaluate, and deploy agents with LangSmith, the agent engineering platform.

- [Unstructured](https://unstructured.io/): Transform complex, unstructured data into clean, structured data. Securely. Continuously. Effortlessly.

- [LanceDB](https://lancedb.com/): From agents to models, from search to training, one platform for all your AI data and workloads

- [FastAPI](https://fastapi.tiangolo.com/): FastAPI framework, high performance, easy to learn, fast to code, ready for production

- [APScheduler](https://apscheduler.readthedocs.io/en/3.x/): Advanced Python Scheduler

- [FastAPI](https://fastapi.tiangolo.com/): FastAPI framework, high performance, easy to learn, fast to code, ready for production

- [FastAPI-Users](https://fastapi-users.github.io/fastapi-users/latest/): Ready-to-use and customizable users management for FastAPI

- [Uvicorn](https://uvicorn.dev/): Uvicorn is an ASGI web server implementation for Python

- [Minio-RAG](https://github.com/dilverse/rag-with-minio): core sample Minio + RAG + Gradio

## Configuration

- DEF_TOP_RERANKER_VECTORS (20): number of embeddings

## Dependencies
To see all dependencies like a tree execute this command:

```shell
$ pipdeptree
```