from langgraph.graph import START, StateGraph, END
from fastapi.responses import JSONResponse
from typing import TypedDict, List, Dict
from langchain_openai import ChatOpenAI
# from duckduckgo_search import ddg
from duckduckgo_search import DDGS
from langgraph.types import Command
from utils.utils import callLLM
from pydantic import BaseModel
from fastapi import FastAPI
from urllib.parse import quote_plus
import requests

app = FastAPI()

class State(TypedDict):
    query: str              # User input
    intent_catogory: str    # Intent of the user 
    key_entities: List[str] # Extracted entities
    confidence_score: str  # how confident LLM is in categorizing the request into a category
    follow_up_questions: List[str] # When info is missing/ambiguous
    web_results: List[Dict]         
    
# Identify the user's intent based on their input.
def get_intent_category(state: State):
    try:
        prompt = """add your prompt here. query: {}"""
        
        prompt = prompt.format(state["query"])
        intent_by_llm = callLLM(prompt)
        print("intent: ", intent_by_llm,"\n\n--------\n\n")
        
        if(intent_by_llm.lower()=="other"):
            return Command(update={"intent_catogory": intent_by_llm}, goto="web_search")
        
        return Command(update={"intent_catogory": intent_by_llm}, goto="get_key_entities")
    except Exception as e:
        return Command(update={"intent_catogory": f"error: {str(e)}"}, goto="get_key_entities")
    
# Extract key entities from the user input based on the provided input and identified intent category.    
def get_key_entities(state: State):
    try:
        prompt = """ add your prompt here. query: {}, intent_category: {}"""
        
        prompt = prompt.format(state["query"], state["intent_catogory"])
        key_entities = callLLM(prompt)
        print("key_entities: ", key_entities,"\n\n--------\n\n")
        return Command(update={"key_entities": key_entities}, goto="isInfoMissing")
    except Exception as e:
        return Command(update={"key_entities": [f"error: {str(e)}"]}, goto="isInfoMissing")

# Generate a confidence score indicating how certain the LLM is in identifying the intent category.
def get_confidence_score(state: State):
    try:
        prompt = """add your prompt here. query: {}, intent_category:{}"""
        
        prompt = prompt.format(state["query"], state["intent_catogory"])
        confidence_score = callLLM(prompt)
        print("confidence_score: ", confidence_score,"\n\n--------\n\n")
        return Command(update={"confidence_score": confidence_score}, goto=END)
    except Exception as e:
        return Command(update={"confidence_score": f"error: {str(e)}"}, goto=END)


# If any required information is missing from the user input, generate appropriate follow-up questions to gather the missing details.
def isInfoMissing(state: State):
    try:
        prompt = """add your prompt here. intent_category: {}, intent_category: {}, query: {}, key_entities: {}"""
        prompt = prompt.format(state["intent_catogory"], state["intent_catogory"], state["query"], state["key_entities"])
        isMissing = callLLM(prompt)
        print("isMissing: ", isMissing,"\n\n--------\n\n")
        
        return Command(update={"follow_up_questions": isMissing}, goto="get_confidence_score")
    except Exception as e:
        return Command(update={"follow_up_questions": [f"error: {str(e)}"]}, goto="get_confidence_score")

# Enable web search functionality for handling non-standard or out-of-scope user requests.
def web_search(state: State):
    
    query = state["query"]
    try:
        with DDGS() as ddgs:
            results = []
            for r in ddgs.text(query, max_results=3):
                if r.get("title") != "EOF":  # Filter out bad results
                    title = str(r.get("title", "No title"))
                    description = str(r.get("body", "No description"))
                    results.append({
                        "title": title[:100] if len(title) > 100 else title,
                        "url": r.get("href", "#"),
                        "description": description[:200] if len(description) > 200 else description
                    })
            
            if not results:
                google_search_url = f"https://www.google.com/search?q={quote_plus(query)}"
                results.append({
                    "title": "Search on Google",
                    "url": google_search_url,
                    "description": f"No results found. Try Google search for: {query}"
                })
            
            
    except Exception as e:
        print(f"Search error: {e}")
        results = [{
            "title": "Search Failed",
            "url": "#",
            "description": "An error occurred during search"
        }]
     
    print("Resp: ", results)
    
    return Command(update={"web_results":results}, goto="get_confidence_score")


    
class RequestBody(BaseModel):
    query: str

graph = StateGraph(State)
graph.add_edge(START, "get_intent_category")
graph.add_node("get_intent_category", get_intent_category)
graph.add_node("get_key_entities", get_key_entities)
graph.add_node("isInfoMissing", isInfoMissing)
graph.add_node("get_confidence_score", get_confidence_score)
graph.add_node("web_search", web_search)

worker = graph.compile()

@app.post("/response")
def chat(request: RequestBody):
    try:
        state_input = {
            "query": request.query,
            "intent_catogory": "",
            "key_entities": [],
            "confidence_score": "",
            "follow_up_questions": []
        }

        state = worker.invoke(state_input)

        if(state["intent_catogory"].lower()=="other"):
                return JSONResponse(
                content={
                    "intent_catogory": state["intent_catogory"],
                    "search_response": state["web_results"],
                    # "key_entities": state["key_entities"],
                    "confidence_score": state["confidence_score"],
                    # "follow_up_questions": state["follow_up_questions"]
                },
                status_code=200,
                media_type="application/json",
            )
            
        return JSONResponse(
            content={
                "intent_catogory": state["intent_catogory"],
                "key_entities": state["key_entities"],
                "confidence_score": state["confidence_score"],
                "follow_up_questions": state["follow_up_questions"]
            },
            status_code=200,
            media_type="application/json",
        )
    except requests.exceptions.RequestException as e:
        return JSONResponse(
            content={"error": {"message": f"Request failed: {str(e)}"}},
            status_code=500
        )
    except Exception as e:
        return JSONResponse(
            content={"error": {"message": f"Unexpected error: {str(e)}"}},
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=9090, reload=True)

