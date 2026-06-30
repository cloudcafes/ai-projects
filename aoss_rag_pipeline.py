import os
import json
import boto3
from opensearchpy import RequestsHttpConnection, AWSV4SignerAuth
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ==========================================
# 1. PLATFORM CONFIGURATION
# ==========================================
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
OPENSEARCH_URL = "https://YOUR-COLLECTION-ID.us-east-1.aoss.amazonaws.com" # UPDATE THIS

# Initialize Amazon Nova Embeddings
embeddings_model = BedrockEmbeddings(
    model_id="amazon.nova-2-multimodal-embeddings-v1:0"
)

# Initialize Claude 4.5 Haiku (Cross-Region Profile)
llm = ChatBedrock(
    model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    model_kwargs={"max_tokens": 1024, "temperature": 0.1}
)

langfuse = Langfuse()

# ==========================================
# 2. OPENSEARCH AUTHENTICATION (IAM)
# ==========================================
def get_opensearch_auth():
    """Uses native AWS IAM roles instead of static passwords"""
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, AWS_REGION, "aoss")
    return auth

def get_vector_store():
    return OpenSearchVectorSearch(
        opensearch_url=OPENSEARCH_URL,
        index_name="it-triage-playbooks",
        embedding_function=embeddings_model,
        http_auth=get_opensearch_auth(),
        timeout=30,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )

# ==========================================
# 3. RUNTIME RAG PIPELINE
# ==========================================
@observe(name="Dynamic-RAG-Support-Triage")
def run_dynamic_triage(ticket_text: str):
    handler = CallbackHandler()
    vector_store = get_vector_store()
    
    print(f"1. Vectorizing user ticket and searching AOSS...")
    # k=2 means we only return the top 2 most mathematically relevant playbooks
    docs = vector_store.similarity_search(ticket_text, k=2)
    retrieved_context = "\n".join([doc.page_content for doc in docs])
    
    print(f"2. Retrieved Context: {retrieved_context.strip()}")
    
    # 3. Fetch Prompt from Governance Registry
    prompt_obj = langfuse.get_prompt("ticket_triage_agent", label="production")
    langchain_prompt = ChatPromptTemplate.from_template(prompt_obj.get_langchain_prompt())
    
    # Notice we are using JsonOutputParser now to strip out the <thinking> tags
    chain = langchain_prompt | llm | JsonOutputParser()
    
    print("3. Executing grounded LLM generation...")
    response = chain.invoke(
        {"context": retrieved_context, "ticket": ticket_text},
        config={"callbacks": [handler]}
    )
    
    return response

# ==========================================
# 4. INITIALIZATION & EXECUTION
# ==========================================
if __name__ == "__main__":
    print("--- Enterprise AI Platform RAG Engine ---\n")
    
    # --- PHASE 1: INGESTION ---
    # In production, this happens asynchronously when playbooks are updated in Git
    print("Ingesting data to OpenSearch Serverless...")
    playbooks = [
        "VPN connectivity issues from outside the US are currently blocked due to a firewall upgrade. Severity: High. Category: Network.",
        "Password resets for Okta require manager approval and 2FA reset. Severity: Low. Category: Identity.",
        "AWS Console access denied errors should be routed to CloudOps with the IAM user ARN. Severity: Medium. Category: Cloud infrastructure.",
        "Blue screen of death on Windows 11 after the recent CrowdStrike update requires safe mode boot. Severity: Critical. Category: Hardware."
    ]
    vector_store = get_vector_store()
    vector_store.add_texts(playbooks) # Converts text to vectors and pushes to AOSS
    
    # --- PHASE 2: INFERENCE ---
    customer_ticket = "I am at a conference in London and my VPN keeps dropping. I can't access any internal sales dashboards."
    
    print("\nProcessing Ticket...")
    final_json = run_dynamic_triage(customer_ticket)
    
    print("\n--- Sanitized Application-Ready JSON ---")
    print(json.dumps(final_json, indent=2))
