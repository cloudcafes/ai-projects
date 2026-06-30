import os
import json
from langfuse.decorators import observe # 1. Import decorator
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


# Initialize global Langfuse client
langfuse = Langfuse()

# 2. Use @observe to automatically trace the entire function call
@observe(name="Support-Ticket-Triage")
def run_triage(ticket_text: str, kb_context: str):
    # Governance: Fetch the production prompt
    prompt_obj = langfuse.get_prompt("ticket_triage_agent", label="production")
    
    langchain_prompt = ChatPromptTemplate.from_template(prompt_obj.get_langchain_prompt())
    chain = langchain_prompt | llm | StrOutputParser()
    
    # 3. Simply invoke the chain; the callback is handled by the decorator context
    # Note: Ensure you still pass the config to capture sub-steps (LLM calls)
    from langfuse.callback import CallbackHandler
    handler = CallbackHandler()
    
    response = chain.invoke(
        {"context": kb_context, "ticket": ticket_text},
        config={"callbacks": [handler]}
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