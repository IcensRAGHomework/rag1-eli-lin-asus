import json
import traceback
import requests

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import (ResponseSchema, StructuredOutputParser)
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.output_parsers import SimpleJsonOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain import hub

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)
llm = AzureChatOpenAI(
    model=gpt_config['model_name'],
    deployment_name=gpt_config['deployment_name'],
    openai_api_key=gpt_config['api_key'],
    openai_api_version=gpt_config['api_version'],
    azure_endpoint=gpt_config['api_base'],
    temperature=gpt_config['temperature']
)

## For format answer

date_schemas = [
    ResponseSchema(
        name="date",
        description="該紀念日的日期",
        type="YYYY-MM-DD"),
    ResponseSchema(
        name="name",
        description="該紀念日的名稱")
]

add_schemas = [
    ResponseSchema(
        name="add",
        description="這是一個布林值，表示是否需要將節日新增到節日清單中。根據問題判斷該節日是否存在於清單中，如果不存在，則為 true；否則為 false",
        type="boolean"),
    ResponseSchema(
        name="reason",
        description="描述為什麼需要或不需要新增節日，具體說明是否該節日已經存在於清單中，以及當前清單的內容。")
]

prompt = ChatPromptTemplate.from_messages([
    ("system","使用台灣語言並回答問題,{format_instructions}"),
    ("human","{question}"),
    MessagesPlaceholder("agent_scratchpad", optional=True)
    ]
)
def get_prompt():
    return prompt

def get_date_result_schemas():
    output_parser = StructuredOutputParser(response_schemas=date_schemas)
    format_instructions = output_parser.get_format_instructions()
    return [
        ResponseSchema(
            name="Result",
            description="一個結果的清單",
            type=format_instructions
        )
    ]

def get_add_result_schemas():
    output_parser = StructuredOutputParser(response_schemas=add_schemas)
    format_instructions = output_parser.get_format_instructions()
    return [
        ResponseSchema(
            name="Result",
            description="只有一筆資料的結果",
            type=format_instructions
        )
    ]

def get_format_instructions(result_schemas: list[ResponseSchema]):
    output_parser = StructuredOutputParser(response_schemas=result_schemas)
    return output_parser.get_format_instructions()

## For generate_hw02
def get_holidays_from_clendarific(year: int, month: int):
    api_key = "WqROpo9g1fRPEpFaSJgMOLXPuWsPXCA3"
    url = f"https://calendarific.com/api/v2/holidays?&api_key={api_key}&country=tw&year={year}&month={month}"
    response = requests.get(url).json().get('response')
    return response

## For gnerate_hw03
history = ChatMessageHistory()
def get_history() -> ChatMessageHistory:
    return history

class GetValue(BaseModel):
    year: int = Field(description="first number")
    month: int = Field(description="second number")

tool = StructuredTool.from_function(
    name="get_value",
    description="當詢問xx年台灣yy月紀念日有哪些?時, xx 為 year, yy 為 month",
    func=get_holidays_from_clendarific,
    args_schema=GetValue
)

def generate_hw01(question):
    date_response_schemas = get_date_result_schemas()
    prompt = get_prompt()
    format_instructions = get_format_instructions(date_response_schemas)
    date_chain = prompt | llm | SimpleJsonOutputParser()
    response = date_chain.invoke({'question': question, "format_instructions": format_instructions})
    return json.dumps(response, ensure_ascii=False)
    
def generate_hw02(question):
    date_response_schemas = get_date_result_schemas()
    format_instructions = get_format_instructions(date_response_schemas)
    prompt = get_prompt()
    tools = [tool]
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    response = agent_executor.invoke({"question": question, "format_instructions": format_instructions}).get('output')
    return json.dumps(SimpleJsonOutputParser().parse(response), ensure_ascii=False)

def generate_hw03(question2, question3):
    prompt = get_prompt()
    prompt.append(MessagesPlaceholder(variable_name="chat_history"))
    tools = [tool]
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    ## first question
    format_instructions = get_format_instructions(get_date_result_schemas())
    agent_with_chat_history.invoke({"question": question2, "format_instructions": format_instructions}).get('output')

    ## second question
    format_instructions = get_format_instructions(get_add_result_schemas())
    response = agent_with_chat_history.invoke({"question": question3, "format_instructions": format_instructions}).get('output')
    return json.dumps(SimpleJsonOutputParser().parse(response), ensure_ascii=False)
    
def generate_hw04(question):
    pass
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response

# print(generate_hw02("2024年台灣10月紀念日有哪些?"))