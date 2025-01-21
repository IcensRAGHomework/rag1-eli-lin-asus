import json
import requests
import base64

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
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from mimetypes import guess_type
from langchain_core.utils.json import parse_json_markdown

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

score_schemas = [
        ResponseSchema(
        name="score",
        description="Integer，用來表示棒球隊的得分",
        type="integer"),
]

def get_prompt():
    return ChatPromptTemplate.from_messages([
    ("system","使用台灣語言並回答問題, 答案必須嚴格遵守格式,{format_instructions}"),
    ("human","{question}"),
    MessagesPlaceholder("agent_scratchpad", optional=True)
    ]
)

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

def get_score_result_schemas():
    output_parser = StructuredOutputParser(response_schemas=score_schemas)
    format_instructions = output_parser.get_format_instructions()
    return [
        ResponseSchema(
            name="Result",
            description="只有一筆資料的結果",
            type=format_instructions
        )
    ]

def get_output_parser(result_schemas: list[ResponseSchema]):
    return StructuredOutputParser(response_schemas=result_schemas)

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

## For generate_hw04
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

image_path = 'baseball.png'
data_url = local_image_to_data_url(image_path)

def get_image_prompt():
    return ChatPromptTemplate.from_messages(
        [
            ("system", "辨識圖片中的文字表格後，使用台灣語言並回答問題, 答案必須嚴格遵守格式,{format_instructions}"),
            (
                "user",
                [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    }
                ],
            ),
            ("human", "{question}")
        ]
    )

def generate_hw01(question):
    date_response_schemas = get_date_result_schemas()
    output_parser = get_output_parser(date_response_schemas)
    prompt = get_prompt()
    format_instructions = output_parser.get_format_instructions()
    date_chain = prompt | llm | output_parser
    response = date_chain.invoke({'question': question, "format_instructions": format_instructions})
    return json.dumps(response, ensure_ascii=False)
    
def generate_hw02(question):
    date_response_schemas = get_date_result_schemas()
    output_parser = get_output_parser(date_response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt = get_prompt()
    tools = [tool]
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    response = agent_executor.invoke({"question": question, "format_instructions": format_instructions}).get('output')
    return json.dumps(parse_json_markdown(response), ensure_ascii=False)

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
    format_instructions = get_output_parser(get_date_result_schemas()).get_format_instructions()
    agent_with_chat_history.invoke({"question": question2, "format_instructions": format_instructions}).get('output')

    ## second question
    format_instructions = get_output_parser(get_add_result_schemas()).get_format_instructions()
    response = agent_with_chat_history.invoke({"question": question3, "format_instructions": format_instructions}).get('output')
    return json.dumps(parse_json_markdown(response), ensure_ascii=False)
    
def generate_hw04(question):
    score_response_schemas = get_score_result_schemas()
    output_parser = get_output_parser(score_response_schemas)
    prompt = get_image_prompt()
    format_instructions = output_parser.get_format_instructions()
    date_chain = prompt | llm | output_parser
    response = date_chain.invoke({'question': question, "format_instructions": format_instructions})
    return json.dumps(response, ensure_ascii=False)
    
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