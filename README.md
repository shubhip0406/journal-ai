# Journal AI (Streamlit + Vertex + Firestore)

## Setup (GCP)
1) Enable APIs: Vertex AI, Cloud Firestore  
2) Create Firestore (Native mode)  
3) Create a Service Account with roles:
   - Cloud Datastore User
   - Vertex AI User
   Create a JSON key and keep it safe.

## Deploy (Streamlit Cloud)
- New app → connect this repo  
- Settings → **Secrets**:

```toml
[gcp]
project_id = "YOUR_PROJECT_ID"
location = "us-central1"
vertex_model = "gemini-1.5-flash"
service_account_json = """{ ...PASTE YOUR FULL SA JSON HERE... }"""


