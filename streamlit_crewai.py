from crewai import  Agent,Task,Crew,Process
from langchain_core.callbacks import BaseCallbackHandler
from typing import TYPE_CHECKING,Any,Dict, Optional
import streamlit as st

from langchain_community.llms import Ollama
#this is how to run stream lit app with crewai streamlit run main.py

dolphin_mistral = Ollama(model ="dolphin-mistral")

avators={"Writer" :"https://cdn-icons-png.flaticon.com/512/320/320336.png",
         "Reviewer":"https://cdn-icons-png.freepik.com/512/9408/9408201.png"}

topic = st.text_input("Enter the topic:")
class MyCustomHandler(BaseCallbackHandler):
    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """ Print out that we are entering a chain. """
        st.session_state.messages.append({"role": "assistant", "content": inputs['input']})
        st.chat_message("assistant").write(inputs["input"])

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """ Print out that we are leaving a chain. """
        st.session_state.messages.append({"role": self.agent_name, "content": outputs['output']})
        st.chat_message(self.agent_name, avator=avators[self.agent_name]).write(outputs["output"])



#define agents

writer = Agent(
    role="Writer",
    goal=f"write a single structure prompt in markdown explaining how a world-class {topic} expert would approch a project",
    backstory= f"You are an AI assistant that writes a single prompt to explaining the {topic} experts from your knowledge base",
    verbose= False, 
    allow_delegation= False,
    llm= dolphin_mistral,
    callbacks=[MyCustomHandler("Reviewer")],
)
reviewer=Agent(
    role="Reviewer",
    goal=f"from your memory,gather relevant information about how an expert at {topic}",
    backstory= f"You are an AI assistant that extracts relevant information to {topic} experts from your knowledge base",
    verbose= False,
    allow_delegation= False,
    llm= dolphin_mistral,
    callbacks=[MyCustomHandler("Writer")],

)



st.title("CrewAI Writting Studio")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role":"assistant","content":"what blog post do you want us to write?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])





#task
gather_info = Task(
    description= f"write a blog post of  {topic} ",
    agent=writer,
    expected_output= f"A clear list of 5 key points related to {topic} experts and how it works",

)

write_prompt = Task(
    description= f"list review comments for improvement from the entire content of blog post {topic} to make it more efficent",
    agent=reviewer,
    expected_output= f"Builtin points about where need to be improved",
)

crew = Crew(
    agents=[writer,reviewer],
    tasks=[gather_info,write_prompt],
    manager_llm=dolphin_mistral,
    verbose=2,
    process= Process.sequential
)
out_put = crew.kickoff()
result = f"## Here is the final Result \n\n {out_put}"
st.session_state.messages.append({"role":"assistant","content":result})
st.chat_message["assistant"].write(result)

'''
researcher_agent=Agent(
    role="researcher",
    goal=f"from your memory,gather relevant information about how an expert at {topic}",
    backstory= f"You are an AI assistant that extracts relevant information to {topic} experts from your knowledge base",
    verbose= True,
    allow_delegation= False,
    llm= dolphin_mistral
)

prompt_agent=Agent(
    role="Prompt engineer",
    goal=f"write a single structure prompt in markdown explaining how a world-class {topic} expert would approch a project",
    backstory= f"You are an AI assistant that writes a single prompt to explaining the {topic} experts from your knowledge base",
    verbose= True, 
    allow_delegation= False,
    llm= dolphin_mistral
)





'''