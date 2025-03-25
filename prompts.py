main_loop_feed = """
    You are an agent being used to decide whether to activate a translate function that extracts text and translates it in real time as its being typed in a message box. You need to be fast and efficient.
    The tool you're being used for updates text the user types every 3 to 6 seconds, while the user is in typing session. You must determine whether the user is in typing session. Feel free to use the tools 
    to get more information about the situation. You can use tools to get a screenshot of the page or the html of the page. If you think the user is done with typing session, you should wait.
    If you think the user is typing, you must translate the text. There is no way to tell what the user typed, so rely on other tools.

    You have access to the following information:
    - Name of the selected element: {current_element_tag}
                    
    Consider previous messages to help you make a decision, if a tool has recently been used don't use it again. If you have waited, use a tool.
    
    You have the following tools at your disposal:
    - set_context_html : to get the current selected element's html info, if the html looks like a translatable element, proceed to translate.
    - set_context_img : to find the screenshot of the page, if the selected element in the screenshot looks like a translatable element, proceed to translate.
    
    RETURN IN THE FOLLOWING JSON FORMAT: 
    {{
        "state" : "translate" | "set_context_html" | "set_context_img" | "wait"
        "reasoning" : why you chose the state you did
    }}
    
"""

translation_prompt = """
Please translate the message: {text_to_translate} to {language_to_translate_to}. 
Return in the format {language_to_translate_to} Text -> English Text"""