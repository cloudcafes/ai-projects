import os
import json
from langfuse import Langfuse
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize Observability and Prompt Registry
langfuse = Langfuse()

# 2. Initialize LLM (Claude 3 Haiku for speed/cost efficiency)
# Note: Claude 3 supports Prompt Caching. In production, placing static 
# context at the beginning of the prompt allows the KV Cache to reuse 
# tokens, drastically reducing costs.
llm = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    client=None, # Automatically uses boto3 credentials
    model_kwargs={
        "max_tokens": 1024,
        "temperature": 0.1 # Low temperature for analytical tasks
    }
)

def run_triage(ticket_text: str, kb_context: str):
    # 3. Governance: Fetch the production prompt dynamically
    prompt_obj = langfuse.get_prompt("ticket_triage_agent", label="production")
    
    # Convert Langfuse prompt to LangChain format
    langchain_prompt = ChatPromptTemplate.from_template(prompt_obj.get_langchain_prompt())
    
    # 4. Create the Chain (Prompt -> LLM -> String Output)
    chain = langchain_prompt | llm | StrOutputParser()
    
    # 5. Execute with Tracing
    # The trace logs latency, token usage, and generation steps
    trace = langfuse.trace(
        name="Support-Ticket-Triage",
        user_id="internal-system",
        metadata={"environment": "production"}
    )
    
    print("Executing LLM chain...")
    response = chain.invoke(
        {"context": kb_context, "ticket": ticket_text},
        config={"callbacks": [trace.get_langchain_handler()]}
    )
    
    return response

if __name__ == "__main__":
    # Simulated Grounding Context (Normally retrieved via RAG / OpenSearch)
    knowledge_base = """
    - VPN connectivity issues from outside the US are currently blocked due to a firewall upgrade. Severity: High. Category: Network.
    - Password resets for Okta require manager approval. Severity: Low. Category: Identity.
    - AWS Console access denied errors should be routed to CloudOps. Severity: Medium. Category: Cloud infrastructure.
    """
    
    customer_ticket = "I am at a conference in London and my VPN keeps dropping. I can't access any internal sales dashboards."
    
    result = run_triage(customer_ticket, knowledge_base)
    print("\n--- Raw LLM Output ---")
    print(result)