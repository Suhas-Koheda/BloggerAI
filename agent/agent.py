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
You have access to tools such as GoogleSearchClient.search and google_search_tool to search internet
When asked to summarize search results about Kotlin Multiplatform:
1. FIRST use the search tool to get current information
return the search tool response in the output field

"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def run_agent_with_formatting(query):
    search_response = agent_executor.invoke({
        "input": f"Search for: {query}. Return full results."
    })

    print(search_response)
    formatting_prompt = f"""
    Here are search results about {query}. Create 5 perfect 5-line summaries 
    following the required format. Only include official sources.
    3. Each summary MUST follow this exact format:
   [1] Title and source
   [2] Key point 1
   [3] Key point 2  
   [4] Key point 3
   [5] Link and date
    Search results:
    {search_response['output']}
    
    Your formatted summaries:
    """

    formatted_response = llm.invoke(formatting_prompt)
    return formatted_response.content

if __name__ == '__main__':
    query = "latest official Artificial Intelligence, models, tools, companies, any other AI news only released it can be any small news also "
    result = run_agent_with_formatting(query)
    print("\nFormatted Results:")
    print(result)