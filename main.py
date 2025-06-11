import base64
import os
import httpx
import requests
from fastapi import FastAPI, Depends, HTTPException, status
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI  # For chat models
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Your OpenAI API Key
# Make sure to set OPENAI_API_KEY environment variable or replace this
openai_api_key = os.getenv("OPENAI_API_KEY")

# Custom Proxy Settings
CUSTOM_PROXY_URL = os.getenv("CUSTOM_PROXY_URL")  # Replace with your URL
SSL_CERTIFICATE_PATH = os.getenv("SSL_CERT")  # Replace with your certificate path
PROXY_USERNAME = os.getenv("PROXY_USERNAME")
PROXY_PASSWORD = os.getenv("PROXY_PASSWORD")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Tool Definitions ---

@tool()
def call_mcp_server(query: str) -> str:
    """
   Use this tool to answer user questions by querying the MCP GraphQL server.

   The input is a valid GraphQL query string. The tool will send this query to the MCP server.

   Example:
   User question: "Tell me about threads on propertyid 1234"
   You should first use GraphQL introspection to understand the schema, then form a valid GraphQL query like:
   `query { property(id: "1234") { threads { id title } } }` and pass it to this tool.
   """

    print(f"Calling MCP server with query: {query}")
    api_url = f"{MCP_SERVER_URL}"
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "query": query
    }
    try:
        response = requests.post(api_url, headers=headers, json=payload, auth=None) # Changed to POST
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        mcp_response_data = response.json()
        return str(mcp_response_data)

    except requests.exceptions.RequestException as e:
        print(f"Error calling MCP server: {e}")
        return f"Error communicating with MCP server: {e}"

@tool()
def get_mcp_graphql_schema() -> str:
    """
    Use this tool to get the GraphQL schema from the MCP server using introspection.
    This should be called first to understand the available queries and types.
    """
    print("Performing GraphQL introspection on MCP server...")
    api_url = f"{MCP_SERVER_URL}"
    headers = {
        "Content-Type": "application/json",
    }
    introspection_query = """
        query IntrospectionQuery {
            __schema {
                queryType { name }
                mutationType { name }
                subscriptionType { name }
                types {
                    ...FullType
                }
                directives {
                    name
                    description
                    locations
                    args {
                        ...InputValue
                    }
                }
            }
        }

        fragment FullType on __Type {
            kind
            name
            description
            fields(includeDeprecated: true) {
                name
                description
                args {
                    ...InputValue
                }
                type {
                    ...TypeRef
                }
                isDeprecated
                deprecationReason
            }
            inputFields {
                ...InputValue
            }
            interfaces {
                ...TypeRef
            }
            enumValues(includeDeprecated: true) {
                name
                description
                isDeprecated
                deprecationReason
            }
            possibleTypes {
                ...TypeRef
            }
        }

        fragment InputValue on __InputValue {
            name
            description
            type { ...TypeRef }
            defaultValue
        }

        fragment TypeRef on __Type {
            kind
            name
            ofType {
                kind
                name
                ofType {
                    kind
                    name
                    ofType {
                        kind
                        name
                        ofType {
                            kind
                            name
                            ofType {
                                kind
                                name
                                ofType {
                                    kind
                                    name
                                    ofType {
                                        kind
                                        name
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    """
    payload = {
        "query": introspection_query
    }
    try:
        response = requests.post(api_url, headers=headers, json=payload, auth=None)
        response.raise_for_status()
        schema_data = response.json()
        return str(schema_data)
    except requests.exceptions.RequestException as e:
        print(f"Error during introspection: {e}")
        return f"Error performing introspection: {e}"

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
    access_token = base64.b64encode(f"{PROXY_USERNAME}:{PROXY_PASSWORD}".encode("ascii")).decode("ascii")

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

    tools = [call_mcp_server, get_mcp_graphql_schema]

    # Define the prompt template
    prompt = PromptTemplate.from_template("""
    You are an AI assistant that interacts with a GraphQL server. 
    Before attempting to query the server, you MUST use the 'get_mcp_graphql_schema' tool to understand the available schema. 
    Once you have the schema, use it to formulate precise GraphQL queries for the 'call_mcp_server' tool.

    Question: {input}
    {agent_scratchpad}
    """)

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor


# --- FastAPI Setup ---
# Pydantic model for the response structure
class ResponseModel(BaseModel):
    response: str

@app.post("/generate_response", response_model=ResponseModel)
async def generate_response(
        prompt: str,
        agent = Depends(get_langchain_chain)
):
    print(f"Received prompt: {prompt}")
    try:
        print(f"Hi here")
        response = agent.invoke({"input": prompt})
        return {"response": str(response)}
    except Exception as e:
        print(f"Error during response generation: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while generating the response")


# --- Run the application (using Uvicorn) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)


