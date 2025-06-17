from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from search.google_search_client import GoogleSearchClient

tools = [
    Tool(
        name="google_search_tool",
        func=GoogleSearchClient.search,
        description="Useful for when you need to answer questions about current events or recent information"
    )
]

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    max_output_tokens=2000
)

system_prompt = """You are an expert research assistant that specializes in summarizing technical news. 
You have access to tools to search the internet for current information.

When asked to summarize search results:
1. FIRST use the search tool to get current information
2. THEN analyze the raw results to create perfect 5-line summaries
3. Each summary MUST follow this exact format:
   [1] Title and source
   [2] Key point 1
   [3] Key point 2  
   [4] Key point 3
   [5] Link and date

Focus only on official sources and recent news (June 2025).
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def run_agent_with_formatting(query):
    # First get search results
    search_response = agent_executor.invoke({
        "input": f"Search for: {query} and return full results from June 2025 only."
    })

    # Get the raw output from the search
    raw_search_output = search_response['output']

    # Format the summaries using the LLM
    formatting_prompt = f"""
    Analyze these search results about {query} and create 3 perfect 5-line summaries 
    following the required format. Only include official sources from June 2025.
    
    Instructions:
    1. Extract information from the raw search results below
    2. Create exactly 3 summaries
    3. Each summary must have 5 lines in this format:
       [1] Title and source
       [2] Key point 1
       [3] Key point 2  
       [4] Key point 3
       [5] Link and date (must be June 2025)
    
    Raw search results:
    {raw_search_output}
    
    Your formatted summaries:
    """

    formatted_response = llm.invoke(formatting_prompt)
    return formatted_response.content

if __name__ == '__main__':
    query = "latest official Artificial Intelligence news including models, tools, companies, and other AI developments released in June 2025"
    print(f"\nSearching for: {query}\n")
    result = run_agent_with_formatting(query)
    print("\nFormatted Results:")
    print(result)