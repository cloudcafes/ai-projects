import os
import json
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler  # Correct path for v4.12.0
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize LLM (Claude 3 Haiku)
llm = ChatBedrock(
    model_id="anthropic.claude-haiku-4-5-20251001-v1:0",
    client=None,
    model_kwargs={"max_tokens": 1024, "temperature": 0.1}
)

# 2. Initialize Langfuse client (only once)
langfuse = Langfuse()

@observe(name="Support-Ticket-Triage")
def run_triage(ticket_text: str, kb_context: str):
    # Fetch prompt from Langfuse
    prompt_obj = langfuse.get_prompt("ticket_triage_agent", label="production")
    langchain_prompt = ChatPromptTemplate.from_template(prompt_obj.get_langchain_prompt())
    chain = langchain_prompt | llm | StrOutputParser()
    
    # Create the CallbackHandler (now from langfuse.langchain)
    handler = CallbackHandler()
    
    response = chain.invoke(
        {"context": kb_context, "ticket": ticket_text},
        config={"callbacks": [handler]}
    )
    return response

if __name__ == "__main__":
    knowledge_base = """
    - VPN connectivity issues from outside the US are currently blocked due to a firewall upgrade. Severity: High. Category: Network.
    - Password resets for Okta require manager approval. Severity: Low. Category: Identity.
    - AWS Console access denied errors should be routed to CloudOps. Severity: Medium. Category: Cloud infrastructure.
    """
    customer_ticket = "I am at a conference in London and my VPN keeps dropping. I can't access any internal sales dashboards."
    result = run_triage(customer_ticket, knowledge_base)
    print("\n--- Raw LLM Output ---")
    print(result)