import base64
import os
import httpx
import requests
from fastapi import FastAPI, Depends, HTTPException, status
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI  # For chat models
from pydantic import BaseModel
from langchain.agents import initialize_agent, AgentType

# --- Configuration ---
# Your OpenAI API Key
# Make sure to set OPENAI_API_KEY environment variable or replace this
openai_api_key = os.getenv("OPENAI_API_KEY")

# Custom Proxy Settings
CUSTOM_PROXY_URL = f"${CUSTOM_PROXY_URL}"  # Replace with your URL
SSL_CERTIFICATE_PATH = f"${SSL_CERT}"  # Replace with your certificate path
PROXY_USERNAME = f"${PROXY_USERNAME}"
PROXY_PASSWORD = f"${PROXY_PASSWORD}"
MCP_SERVER_URL = f"${MCP_SERVER_URL}"

# --- Tool Definitions ---
# --- Tool Definitions ---

#Testing client connection


@tool()
def call_mcp_server(query: str) -> str:
    """
   Use this tool to answer user questions by querying the MCP GraphQL server.

   The input is the user's natural language question. You should convert it to a GraphQL query
   internally before calling the MCP server.

   Example:
   User question: "How many messages were received for property 1234 in January 2024?"
   Internally, you should convert this to a valid GraphQL query and send it to the MCP server.
   """

    print(f"Calling MCP server with query: {query}")
    api_url = f"{MCP_SERVER_URL}"
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "query": query  # Pass the LLM's choosen query directly
    }
    try:
        response = requests.get(api_url, headers=headers, json=payload, auth=None) # Include authentication if needed
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        # Process the response from the MCP server
        mcp_response_data = response.json()  # Assuming the response is JSON

        # Convert the response data to a string for the LLM
        return str(mcp_response_data)

    except requests.exceptions.RequestException as e:
        print(f"Error calling MCP server: {e}")
        return f"Error communicating with MCP server: {e}"

# --- Dependency for OpenAI Client with Proxy ---

def get_openai_client() -> httpx.Client:
    """
    Provides an httpx.Client configured with the custom proxy and basic auth.
    """
    http_client = httpx.Client(
        proxy=CUSTOM_PROXY_URL,
        transport=httpx.HTTPTransport(verify=SSL_CERTIFICATE_PATH),
        auth=httpx.BasicAuth(PROXY_USERNAME, PROXY_PASSWORD),
    )
    return http_client

# --- LangChain Setup (as a dependency) ---

def get_langchain_chain(http_client: httpx.Client = Depends(get_openai_client)):
    """
    Provides the LangChain chain configured with the OpenAI model and tools.
    """
    access_token = base64.b64encode(f"{PROXY_USERNAME}:{PROXY_PASSWORD}".encode('ascii')).decode('ascii')

    custom_headers = {
        "x-client-app": PROXY_USERNAME
    }
    llm = ChatOpenAI(
        openai_api_key = access_token,
        model="gpt-4.1-mini-2025-04-14",
        http_client=httpx.Client(verify=SSL_CERTIFICATE_PATH),
        base_url=CUSTOM_PROXY_URL,
        default_headers=custom_headers,
    )

    agent = initialize_agent(
        tools=[call_mcp_server],
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
    )
    return agent


# --- FastAPI Setup ---
# Pydantic model for the response structure
class ResponseModel(BaseModel):
    response: str

@app.post("/generate_response", response_model=ResponseModel)
async def generate_response(
        prompt: str,
        # chain = Depends(get_langchain_chain),
        agent = Depends(get_langchain_chain)
):
    print(f"Received prompt: {prompt}")
    try:
        # print("Test mcp")
        print(f"Hi here")
        # response = chain.invoke({"input": prompt})

        response = agent.invoke({"input": prompt})
        # Ensure the response is a string before returning
        return {"response": str(response)}
    except Exception as e:
        print(f"Error during response generation: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while generating the response")


# --- Run the application (using Uvicorn) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
