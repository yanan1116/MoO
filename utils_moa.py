import os
import json
import time
import requests
import openai
import copy,tiktoken
from retrying import retry
from loguru import logger
from colorama import Fore,init
init(autoreset=True)

DEBUG = int(os.environ.get("DEBUG", "0"))
from together import Together
client = Together()


def print_message(messages):
    for i in messages:
        print(i['role'], '===>')
        print(i['content'])




review_prompt = """review the answer and response to the given question. 
give your concise and brief feedback and opinion on the answer, no more than 200 words, 
and check if there are any issues within the answer.
If there is no issue or the answer is correct, just write the answer is correct.
"""

def inject_references_to_messages(
    messages,
    references,
    reference_models
):

    # system = "Take these following responses and answers from other agents into consideration for your response."

    system = "You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.\n\nResponses from models:"
    for i, reference in enumerate(references):
        system += f"\nresponse from agent #{i+1}: {reference}"
        
        # inject feedback to each reference
        if reference_models:
            print(Fore.RED + "inject feedback to each reference")
            prompt_feedback = messages + [{'role':'assistant', 'content':reference}, {'role':'user', 'content':review_prompt}]

            for j, reference_model in enumerate(reference_models):
                print(Fore.GREEN + 'illicit feedback from {}'.format(reference_model) + '===>' )
                response = client.chat.completions.create(model=reference_model, 
                                                         messages=prompt_feedback, 
                                                         max_tokens=512, 
                                                         temperature=0)
                feedback = response.choices[0].message.content.strip()
                try:
                    print(Fore.CYAN + feedback)
                except:
                    pass
                system += f"\n \tagent #{j+1}'s feedback and comment on the response of agent #{i+1}: {feedback}\n"
     
        system +=  '\n-------------------------------------------\n'
    
    messages_copy = copy.deepcopy(messages)
    assert messages_copy[0]['role'] == 'system'
    # messages_copy[0]['content'] += system
    # return messages_copy
    return messages + [ {"role": "user", "content": system}]


# @retry(stop_max_attempt_number=5, wait_random_min=1, wait_random_max=5)
def generate_with_references(
    model,
    messages,
    references,
    max_tokens,
    temperature,
    reference_models ):

    if len(references) > 0:
        # print(Fore.CYAN + 'before injection===>')
        # print_message(messages)

        messages = inject_references_to_messages(messages, references, reference_models)

        # print(Fore.CYAN + 'after injection<===')
        # print_message(messages)

    # print(Fore.BLUE +  'illicit response from {}'.format(model))
    response = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
    output = response.choices[0].message.content
    return output.strip()


