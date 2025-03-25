from langgraph.graph import MessagesState, START, END, StateGraph
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
import os
from playwright.async_api import Page
from typing_extensions import TypedDict
import asyncio
from prompts import main_loop_feed, translation_prompt
import json
import base64

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
        self.add_edge("translate_message", "wait")
        self.add_edge("wait", "set_selected_element")
        # additional attributes
        self.page = None
        self.language_to_translate_to = "Arabic"

    def set_page(self, page : Page):
        self.page = page

    async def check_selected_element(self):
        """ Checks and returns the selected element has changed """
        if self.page and isinstance(self.page, Page):
            selected_element = await self.page.evaluate("document.activeElement?.tagName")
            if str(selected_element).lower() == "input":
                selected_element = await self.page.evaluate("document.activeElement?.className")
        return selected_element

    async def set_selected_element(self, state : RT_State):
        """ Sets the tag name of the element currently selected by the mouse om the selected_element field of the state """
        if self.page and isinstance(self.page, Page):
            selected_elem = await self.page.evaluate("document.activeElement?.tagName")
            if str(selected_elem).lower() == "input":
                selected_elem = await self.page.evaluate("document.activeElement?.className")
            state['selected_element'] = selected_elem
        print("page: ", self.page)
        print("set selected element: ", selected_elem)
        return state

    @tool
    async def set_context_html(self, state : RT_State):
        """ Sets the inner html of the element currently selected by the mouse on the context_html field of the state """
        print("get context html")
        if self.page and isinstance(self.page, Page):
            context_html = await self.page.evaluate("document.activeElement?.outerHTML")
            print("context_html: ", context_html)
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
                You need to decide if the selected element is a message box. Do use the tools at hand in case you feel unsure what the selected element is. 
                You have may have access to the following information:
                
                Name of the selected element: {selected_element}
                    
                You have the following tools at your disposal:
                set_context_html - to find the html of the page
                set_context_img - to find the screenshot of the page
                    
                If you think the selected element is a text box, return translate_message
                If you think you need to use a tool, return the name of the tool
                Otherwise, return wait
                          
                RETURN ONLY ONE WORD: translate_message, set_context_html, set_context_img, or wait
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
        if "input" in state['selected_element'].lower():
            eval_str = "{selected_element} input[type=text]".format(selected_element=state['selected_element'])
        else:
            eval_str = "input.{selected_element}".format(selected_element=state['selected_element'])
        if self.page and isinstance(self.page, Page):
            try:
                await self.page.wait_for_selector(eval_str, state='visible', timeout=5000)
            except:
                print("element not found")
                return state
            try:
                raw_value = await self.page.locator(eval_str).input_value()
                print("raw_value: ", raw_value)
                if raw_value:
                    # handle translating the text
                    text_to_translate = raw_value.split(" -> ")[1] if len(raw_value.split(" -> ")) > 1 else raw_value.split(" -> ")[0]
                    print("text_to_translate: ", text_to_translate)
                    state['messages'].append(
                        HumanMessage(content=translation_prompt.format(
                            text_to_translate=text_to_translate,
                            language_to_translate_to=self.language_to_translate_to
                        ))
                    )
                    translated_text = self.chat.invoke(state['messages'])
                    print("translated_text: ", translated_text.content.strip())
                    state['messages'].append(translated_text)
                    # fill the text box with the translated text
                    await self.page.locator(eval_str).fill(translated_text.content.strip())
            except:
                print("error in translating message")
                return state
            
        # wait for 10 seconds and then check if the selected element has changed
        await asyncio.sleep(10)
        state['selected_element'] = await self.page.evaluate("document.activeElement?.tagName")
        if str(state['selected_element']).lower() == "input":
            state['selected_element'] = await self.page.evaluate("document.activeElement?.className")
        if state['selected_element'] != "INPUT":
            return state

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
        print("------------------------------------------------")
        # keep only the last 3 messages to get historical context
        state['messages'] = state['messages'][-1:]
        # print("Chat so far: ", [message.content + " -> " for message in state['messages']])
        await asyncio.sleep(10)
        return state


class RT_State_v2(TypedDict):
    current_element_tag : str
    context_html : str
    context_img : str
    messages : list[BaseMessage]

class RT_Graph_v2(StateGraph):
    def __init__(self):
        super().__init__(RT_State)
        # add the node
        self.add_node("main_loop", self.main_loop)
        # add the chat
        self.chat = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        # self.chat = self.chat.bind_tools([self.set_context_html, self.set_context_img])
        # add the edges
        self.add_edge(START, "main_loop")
        # additional attributes
        self.cur_state = "translate" # options are tool_call|translate|wait
        self.page = None
        self.language_to_translate_to = "Bengali"


    def set_page(self, page : Page):
        self.page = page

    async def main_loop(self, state : RT_State_v2):
        """ Main loop for the agent """
        while True:
            # prepare data to feed to the LLM
            state['current_element_tag'] = await self.page.evaluate("document.activeElement?.tagName")

            # feed all data to the LLM
            state['messages'].append(SystemMessage(content=main_loop_feed.format(
                current_element_tag=state['current_element_tag'] 
            )))
            ai_result = self.chat.invoke(state['messages'])
            print("ai_result: ", ai_result.content)
            count = 0
            resp = None
            while count < 3:
                try:
                    if len(ai_result.content) > 0:
                        resp = json.loads(ai_result.content)
                        # print("resp: ", resp)
                        break
                except:
                    print("error in parsing the response")
                    count += 1
                    ai_result = self.chat.invoke(state['messages'])
                    continue
            if count == 3:
                print("failed to parse the response")
                await asyncio.sleep(5)
                continue
            
            # execute the actions the LLM choses and update the state
             
            if resp:
                if resp['state'] == "set_context_html":
                    state = await self.set_context_html(state)
                elif resp['state'] == "set_context_img":
                    state = await self.set_context_img(state)
                elif resp['state'] == "translate":
                    state = await self.translate_message(state)
                elif resp['state'] == "wait":
                    await asyncio.sleep(6)
                    state['messages'].append(HumanMessage(content="Waited for 6 seconds"))
            else:
                await asyncio.sleep(3)
                state['messages'].append(HumanMessage(content="Waited for 3 seconds"))
            
            state = self.optimize_message_chain(state)

    def optimize_message_chain(self, state : RT_State_v2):
        """ Optimizes the message chain by using the last 6 messages """
        # remove duplicate messages
        state['messages'] = state['messages'][-6:]
        return state

    
    async def set_context_html(self, state : RT_State_v2):
        """ Sets the inner html of the element currently selected by the mouse on the context_html field of the state """
        print("get context html")
        if self.page and isinstance(self.page, Page):
            context_html = await self.page.evaluate("document.activeElement?.outerHTML")
            if len(context_html) > 1000:
                state['messages'].append(HumanMessage(content="The html is too long. The selected element is not a translatable element."))
                return state
            print("context_html: ", context_html)
            state['context_html'] = context_html
            state['messages'].append(HumanMessage(content=f"After running the set_context_html tool, here is the inner html of the element: {context_html}. Remember to translate the text if the element looks like a fillable text element."))
        return state

    
    async def set_context_img(self, state : RT_State_v2):
        """ Sets the image of the element currently selected by the mouse on the context_img field of the state """
        print("get context img")
        if self.page and isinstance(self.page, Page):
            screenshot_bytes = await self.page.screenshot()
            encoded_image = base64.b64encode(screenshot_bytes).decode('utf-8')
            # pass thru an LLM to describe the image
            messages = [
                SystemMessage(content="Describe the selected element in the image, if any element is selected. Also what is the text in the selected element."),
                HumanMessage(content=[{
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                }])
            ]
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
            img_desc = llm.invoke(messages)
            state['context_img'] = img_desc
            print("img_desc: ", img_desc)
            state['messages'].append(HumanMessage(content=f"After running the set_context_img tool, here is the inner html of the element: {img_desc}"))
        return state

    async def translate_message(self, state : RT_State_v2):
        """ Translates the message in the selected element to Spanish """
        print("translate message")
        # find the element selected by the mouse
        if str(state['current_element_tag']).lower() == "input":
            # get class name of the element
            class_name = await self.page.evaluate("document.activeElement?.className")
            eval_str = f"input.{class_name}"
        else:
            eval_str = f"{state['current_element_tag'].lower()} input[type='text']"
        if self.page and isinstance(self.page, Page):
            # wait for the element to be visible
            try:
                element = await self.page.wait_for_selector(eval_str, state='visible', timeout=5000)
            except:
                print("element not found")
                return state
            # get the text from the element
            # the state "last_translation_text" is used to store the text to fill with. Stored in state for global access outside of try-catch
            try:
                # We should use .input_value() on the element handle instead
                raw_value = await element.input_value()
                print("raw_value: ", raw_value)
                if raw_value:
                    # handle translating the text
                    text_to_translate = raw_value.split(" -> ")[1] if len(raw_value.split(" -> ")) > 1 else raw_value.split(" -> ")[0]
                    print("text_to_translate: ", text_to_translate)
                    state['last_translation_text'] = text_to_translate
            except:
                print("error in getting the text from the element")
                return state
            # translate the text
            try:
                translated_text = self.chat.invoke([
                    HumanMessage(
                        content=translation_prompt.format(
                            text_to_translate=state['last_translation_text'],
                            language_to_translate_to=self.language_to_translate_to
                    ))
                ])
                state['last_translation_text'] = translated_text.content.strip()
                print("translated_text: ", translated_text.content.strip())
                # update the messages list
                state['messages'].append(
                    HumanMessage(content=f"Here is the translated text: {translated_text.content.strip()} from {text_to_translate}")
                )
            except Exception as e:
                print("error in translating the text: ", e)
                return state
            # fill the text in the element
            try:
                await element.fill(state['last_translation_text'])
            except:
                print("error in filling the text in the element")
                return state
            # wait for 3 seconds then return
            await asyncio.sleep(3)
            return state        