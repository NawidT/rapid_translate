from langgraph.graph import MessagesState, START, END, StateGraph
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
import os
from playwright.async_api import Page
from typing_extensions import TypedDict
import asyncio

class RT_State(TypedDict):
    selected_element : str
    context_html : str
    context_img : str
    messages : list[BaseMessage]

class RT_Graph(StateGraph):
    def __init__(self):
        super().__init__(RT_State)
        # add the nodes
        self.add_node("set_selected_element", self.set_selected_element)
        self.add_node("set_context_html", self.set_context_html)
        self.add_node("set_context_img", self.set_context_img)
        self.add_node("decide_if_selected_is_message_box", self.decide_if_selected_is_message_box)
        self.add_node("translate_message", self.translate_message)
        self.add_node("wait", self.handle_wait)
        # add the chat
        self.chat = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        self.chat = self.chat.bind_tools([self.set_context_html, self.set_context_img])
        # add the edges
        self.add_edge(START, "set_selected_element")
        self.add_edge("set_selected_element", "decide_if_selected_is_message_box")
        self.add_edge("set_context_html", "decide_if_selected_is_message_box")
        self.add_edge("set_context_img", "decide_if_selected_is_message_box")
        self.add_conditional_edges(
            "decide_if_selected_is_message_box",
            self.cond_edge_is_messbox,
        )
        self.add_edge("translate_message", "set_selected_element")
        self.add_edge("wait", "set_selected_element")
        # additional attributes
        self.page = None
        self.language_to_translate_to = "Spanish"


    def set_page(self, page : Page):
        self.page = page

    async def set_selected_element(self, state : RT_State):
        """ Sets the tag name of the element currently selected by the mouse om the selected_element field of the state """
        if self.page and isinstance(self.page, Page):
            selected_elem = await self.page.evaluate("document.activeElement?.tagName? || ''")
            state['selected_element'] = selected_elem
        print("page: ", self.page)
        print("set selected element: ", selected_elem)
        return state

    @tool
    async def set_context_html(self, state : RT_State):
        """ Sets the inner html of the element currently selected by the mouse on the context_html field of the state """
        print("get context html")
        if self.page and isinstance(self.page, Page):
            context_html = await self.page.evaluate("document.activeElement?.outerHTML || ''    ")
            state['context_html'] = context_html
            state['messages'].append(HumanMessage(content=f"Here is the inner html of the element: {context_html}"))
        return state

    @tool
    def set_context_img(self, state : RT_State):
        """ Sets the image of the element currently selected by the mouse on the context_img field of the state """
        print("get context img")
        if self.page and isinstance(self.page, Page):
            context_img = self.page.screenshot()
            # pass thru an LLM to describe the image
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
            img_desc = llm.invoke(context_img)
            state['context_img'] = img_desc
            print("img_desc: ", img_desc)
            state['messages'].append(HumanMessage(content=f"Here is the description of the image of the element: {img_desc}"))
        return state
        
    def decide_if_selected_is_message_box(self, state : RT_State):
        """ Decides if the selected element is a message box using an LLM with access to tools (set_context_html, set_context_img) """
        state['messages'].append(
            SystemMessage(content="""
                Is the selected element a message box?, You are an agent being used to translate a message as its being typed in a message box.
                You need to decide if the selected element is a message box. You have may have access to the following information:
                Name of the selected element: {selected_element}
                    
                You have the following tools at your disposal:
                set_context_html - to find the html of the page
                set_context_img - to find the screenshot of the page
                    
                If you think the selected element is a text box, return translate_message 
                If you think you need to use a tool, return the name of the tool
                Otherwise, return wait
        """.format(selected_element=state['selected_element']))
        )
        state['messages'].append(
            HumanMessage(content="Is the selected element a message box?")
        )
        ans = self.chat.invoke(state['messages'])
        print("ans: ", ans.content.strip())
        state['messages'].append(ans)
        return state
    
    async def translate_message(self, state : RT_State):
        """ Translates the message in the selected element to Spanish """
        print(state['selected_element'])
        eval_str = "document.getElementsByTagName('{selected_element}')[0]".format(selected_element=state['selected_element'])
        if self.page and isinstance(self.page, Page):
            raw_value = await self.page.evaluate(eval_str + ".value")
            print("raw_value: ", raw_value)
            text_to_translate = raw_value.split(" -> ")[0]
            print("text_to_translate: ", text_to_translate)
            state['messages'].append(
                HumanMessage(content=f"Please translate the message: {text_to_translate} to {self.language_to_translate_to}. 
                             Keep the format of english_text -> spanish_text")
            )
            translated_text = self.chat.invoke(state['messages'])
            print("translated_text: ", translated_text.content.strip())
            state['messages'].append(translated_text)
            self.page.locator(eval_str).fill(translated_text.content.strip())
        return state

    def cond_edge_is_messbox(self, state : RT_State):
        if state['messages'][-1].tool_calls:
            print("tool call")
            print(state['messages'][-1].tool_calls)
            return state['messages'][-1].tool_calls[0].name
        elif state['messages'][-1].content.strip() == "translate_message":
            print("THIS IS A MESSAGE BOX")
            return "translate_message"
        else:
            print("no tool call")
            return "wait"
        
    async def handle_wait(self, state : RT_State):
        print("waiting")
        # keep only the last 4 messages to get historical context
        state['messages'] = state['messages'][-4:]
        # print("Chat so far: ", [message.content + " -> " for message in state['messages']])
        await asyncio.sleep(10)
        return state