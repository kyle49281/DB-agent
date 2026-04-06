from typing import Annotated, TypedDict
import psycopg2
import os
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
# 1. Define the State (The "Whiteboard")
class AgentState(TypedDict):
    # add_messages tells LangGraph to append new messages instead of overwriting
    messages: Annotated[list, add_messages]

os.environ["TAVILY_API_KEY"] = "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"
search_tool = TavilySearch(max_results=2)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

system_prompt = (
    
    "You are a data analyst.DO NOT explain your process.\n "
    "If you are unsure which table to use or how columns are named, call the discover_database_structure tool first. Then, use execute_sql to answer the user.\n"
    "When asked about data, follow this strict SEARCH PROTOCOL:\n"
    "1. FIRST: Use 'list_tables' and 'execute_sql' to find structured data (like efficiency ratings or costs).\n"
    "2. SECOND: If the answer isn't in a table, use 'local_docs_search' to check kyle's notes.\n"
    "3. THIRD: Use 'web_search' only for external benchmarks or 2026 industry news.\n\n"
    "DO NOT mention 'Search Protocols' or 'JSON'. "
    "If you need information, call the appropriate tool immediately. "
    "Once you have the tool output, provide the answer in a bulleted list."
)

@tool
def list_tables():
    """Returns a list of all tables in the database so you know what is available."""
    conn = psycopg2.connect(dbname="ai_lab", user="kyle", host="localhost")
    cur = conn.cursor()
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    tables = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return f"Available tables: {', '.join(tables)}"
@tool
def execute_sql(query: str):
    """
    Executes SQL commands against the PostgreSQL 'ai_lab' database.
    Supports SELECT (reading), CREATE (building), INSERT, and UPDATE.
    """
    print(f"🚀 Executing SQL: {query}")
    try:
        conn = psycopg2.connect(dbname="ai_lab", user="kyle", host="localhost")
        
        # KEY FIX 1: Set autocommit to True so changes are saved immediately
        conn.autocommit = True 
        
        cur = conn.cursor()
        cur.execute(query)
        
        # KEY FIX 2: Only fetch rows if the query actually returns data (like SELECT)
        if cur.description: 
            rows = cur.fetchall()
            result = f"Query Results: {str(rows)}"
        else:
            # For CREATE, INSERT, etc., return a success message
            result = f"Success: Command executed. Rows affected: {cur.rowcount}"
            
        cur.close()
        conn.close()
        return result
        
    except Exception as e:
        # Safety: ensure connection closes even on failure
        if 'conn' in locals() and conn:
            conn.close()
        return f"❌ SQL Error: {e}. Check your syntax or table names."
@tool
def run_math_analysis(code: str):
    """Executes python code. Define 'result = ...' for the final answer."""
    print(f"🤓crunching numbers")
    
    local_vars = {}
    try:
        exec(code, {"__builtins__": __builtins__}, local_vars)
        return str(local_vars.get('result', "Success"))
    except Exception as e:
        return f"Math Error: {e}"
@tool
def local_docs_search(query: str):
    """Use this tool to find technical context, project updates, and manual entries for the Helios project. Check here if the SQL tables are missing details or if you need to explain 'why' a number is the way it is."""

    print(f"🔍 [DB QUERY] Searching for: {query}")
    
    try:
        # 1. Convert the query into a vector
        query_vector = embeddings.embed_query(query)

        # 2. Connect to Postgres
        conn = psycopg2.connect(dbname="ai_lab", user="kyle", host="localhost")
        cur = conn.cursor()

        # 3. Perform Vector Similarity Search (<=> is Cosine Distance)
        cur.execute("""
            SELECT content FROM doc_chunks 
            ORDER BY embedding <=> %s::vector 
            LIMIT 3;
        """, (query_vector,))
        
        results = cur.fetchall()
        cur.close()
        conn.close()

        # 4. Join the results into a single string for the AI to read
        context = "\n---\n".join([r[0] for r in results])
        return context if context else "No relevant local documents found."
        
    except Exception as e:
        return f"Database Error: {e}"

@tool
def discover_database_structure():
    """
    Queries the database to find all table names and their column names. 
    Use this to understand the schema before writing a query.
    """
    print(f'looking for all tables and their column names...')
    conn = psycopg2.connect(dbname="ai_lab", user="kyle", host="localhost")
    cur = conn.cursor()
    
    # This query finds all tables and their columns in the 'public' schema
    cur.execute("""
        SELECT table_name, column_name 
        FROM information_schema.columns 
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # Format it into a clean list for the AI
    schema_info = ""
    for table, column in rows:
        schema_info += f"Table: {table} | Column: {column}\n"
    
    return schema_info

tools = [
    search_tool, 
    local_docs_search, # Vector search
    list_tables,       # SQL helper 1
    execute_sql,        # SQL helper 2
    discover_database_structure
]

# 2. Setup the LLM and Tools
llm = ChatOllama(model="llama3.1:8b", temperature=0).bind_tools(tools)

# 3. Define the Nodes
def call_model(state: AgentState):
    """The brain: decides what to do next."""
    prompt = SystemMessage(content=(
        "You are a strict SQL Data Analyst. "
        "When you need a tool, you MUST call it directly. "
        "DO NOT write any conversational text, explanations, or 'I will use...' sentences. "
        "Your response should ONLY be the tool call. "
        "Only after you receive the tool's output should you provide a final summary."
    ))
    messages = [prompt] + state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

def save_to_history(state: AgentState):
    """Saves the final interaction to the database audit log."""
    last_message = state["messages"][-1].content
    # Find the original user question (the first message in the list)
    user_prompt = state["messages"][0].content
    print(last_message)
    print(user_prompt)
    try:
        conn = psycopg2.connect(dbname="ai_lab", user="kyle", host="localhost")
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO analysis_results (user_prompt, summary) VALUES (%s, %s)", 
            (user_prompt, last_message)
        )
        conn.commit()
        cur.close()
        conn.close()
        print("💾 [System] Interaction logged to analysis_results.")
    except Exception as e:
        print(f"⚠️ Logging Error: {e}")
    
    return state # Nodes must return the state

# 4. Build the Graph
workflow = StateGraph(AgentState)

# Add our "Workers"
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools)) # Prebuilt helper for tool execution
workflow.add_node("saver", save_to_history)

# Define the Flow
workflow.add_edge(START, "agent")

# The Decision: If model wants tools -> go to tools. Otherwise -> END.

workflow.add_conditional_edges(
    "agent", 
    tools_condition,
    {
        "tools": "tools",  # If tool_calls exist
        "__end__": "saver" # If no tool_calls, go to saver instead of END
    }
)
# After tools run, they MUST go back to the agent to summarize
workflow.add_edge("tools", "agent")
workflow.add_edge("saver", END)
# Compile the App
app = workflow.compile()


def smart_query(user_input):
    inputs = {"messages": [HumanMessage(content=user_input)]}
    
    print("--- 🛰️  Agent Link Established ---")
    
    # .stream() returns a generator that yields updates after each node
    for output in app.stream(inputs, config={"recursion_limit": 20}):
        # 'output' is a dict where keys are node names (e.g., 'agent', 'tools')
        for node_name, state_update in output.items():
            print(f"\n[Node: {node_name}]")
            
            
            # If the agent just spoke, show its thought or tool call
            if node_name == "agent":
                last_msg = state_update["messages"][-1]
                if last_msg.tool_calls:
                    print(f"🛠️  Calling Tools: {[t['name'] for t in last_msg.tool_calls]}")
                else:
                    print(f"📝 Finalizing Response...")

            # If the tools just ran, show a snippet of what they found
            elif node_name == "tools":
                print("📋 Data retrieved from Database/Web.")

    # After the loop finishes, print the final result from the state
    # (The state is preserved in the last 'output')
    final_state = list(output.values())[0] 
    print("\n✅ FINAL ANSWER:")
    print(final_state["messages"][-1].content)

if __name__ == "__main__":
    smart_query("list out all table and columns")

'''
"create a new table called 'users'."
"list out all table and columns"
'''
