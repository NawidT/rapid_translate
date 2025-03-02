from langgraph.graph import MessagesState, START, END, StateGraph
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
import os
from playwright.sync_api import Page

class RT_State(TypedDict):
    page : object
    selected_element : str
    context_html : str
    context_img : str
    messages : list[BaseMessage]

class RT_Graph(StateGraph):
    def __init__(self):
        super().__init__()
        # add the nodes
        self.add_node("set_selected_element", self.set_selected_element)
        self.add_node("set_context_html", self.set_context_html)
        self.add_node("set_context_img", self.set_context_img)
        self.add_node("decide_if_selected_is_message_box", self.decide_if_selected_is_message_box)
        # add the chat
        self.chat = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        self.chat = self.chat.bind_tools([self.set_context_html, self.set_context_img])
        # add the edges
        self.add_edge("set_selected_element", "decide_if_selected_is_message_box")
        self.add_edge("set_context_html", "decide_if_selected_is_message_box")
        self.add_edge("set_context_img", "decide_if_selected_is_message_box")
        self.add_conditional_edges(
            "decide_if_selected_is_message_box",
            self.cond_edge_is_messbox,
        )
        # compile the graph
        self.compile()

    def set_page(self, page : Page):
        self.page = page

    def set_selected_element(self, state : RT_State):
        """ Sets the tag name of the element currently selected by the mouse om the selected_element field of the state """
        print("set selected element")
        if isinstance(state['page'], Page):
            hover_element = state['page'].locator(':hover')
            selected_elem = hover_element.evaluate('e => e.tagName')
            state['selected_element'] = selected_elem
        return state

    @tool
    def set_context_html(self, state : RT_State):
        """ Sets the inner html of the element currently selected by the mouse on the context_html field of the state """
        print("get context html")
        if  isinstance(state['page'], Page):
            context_html = state['page'].evaluate('document.querySelector(":hover").innerHTML')
            state['context_html'] = context_html
            state['messages'].append(SystemMessage(content=f"Here is the inner html of the element: {context_html}"))
        return state

    @tool
    def set_context_img(self, state : RT_State):
        """ Sets the image of the element currently selected by the mouse on the context_img field of the state """
        print("get context img")
        if isinstance(state['page'], Page):
            context_img = state['page'].screenshot()
            # pass thru an LLM to describe the image
            state['context_img'] = context_img
            state['messages'].append(SystemMessage(content=f"Here is the description of the image of the element: {context_img}"))
        return state
        
    def decide_if_selected_is_message_box(self, state : RT_State):
        """ Decides if the selected element is a message box using an LLM with access to tools (set_context_html, set_context_img) """
        state['messages'].append(
            SystemMessage(content="""
                Is the selected element a message box?, You are an agent being used to translate a message as its being typed in a message box.
                You need to decide if the selected element is a message box. You have may have access to the following information:
                Name of the selected element: {selected_element}
                    
                You have the following tools at your disposal:
                set_context_html - to find the html of the page if you don't have it yet
                set_context_img - to find the screenshot of the page if you don't have it yet
                    
                If you think the selected element is a message box, return True. 
                If you think you need to use a tool, return the name of the tool
                Otherwise, return False.
        """.format(selected_element=state['selected_element']))
        )
        state['messages'].append(
            HumanMessage(content="Is the selected element a message box?")
        )
        ans = self.chat.invoke(state['messages'])
        state['messages'].append(ans)
        return state

    def cond_edge_is_messbox(self, state : RT_State):
        if state['messages'][-1].tool_calls:
            print("tool call")
            return state['messages'][-1].tool_calls[0].name
        else:
            print("no tool call")
            return "wait"