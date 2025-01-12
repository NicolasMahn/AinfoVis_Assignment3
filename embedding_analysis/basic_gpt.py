import openai
import tiktoken
from scrt import OPENAI_KEY

WHITE = "\033[97m"
BLUE = "\033[34m"
GREEN = "\033[32m"
ORANGE = "\033[38;5;208m"
PINK = "\033[38;5;205m"
RESET = "\033[0m"

MAX_TOKENS = 128000  # Set the maximum token limit for the model

def count_tokens(prompt: str, model: str = "gpt-4o") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(prompt))

def ask_gpt(prompt: str, role :str="You are a helpful assistant.", temperature=0.2):
    openai.api_key = OPENAI_KEY
    if count_tokens(prompt, model="gpt-4o") > MAX_TOKENS:
        raise ValueError("Prompt exceeds the maximum token limit.")
    response_text = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": role},
            {
                "role": "user",
                "content": prompt,
                "temperature": temperature
            }
        ]
    )
    # print(f"{PINK}ROLE:\n{role}{RESET}")
    # print(f"{BLUE}PROMPT:\n{prompt}{RESET}")
    # print(f"{GREEN}RESPONSE:\n{response_text.choices[0].message.content}{RESET}")
    # print(f"---")

    return response_text.choices[0].message.content

def ask_mini_gpt(prompt: str, role :str="You are a helpful assistant.", temperature=0.2):
    openai.api_key = OPENAI_KEY
    if count_tokens(prompt, model="gpt-4o-mini") > MAX_TOKENS:
        raise ValueError("Prompt exceeds the maximum token limit.")
    response_text = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": role},
            {
                "role": "user",
                "content": prompt,
                "temperature": temperature
            }
        ]
    )
    # print(f"{PINK}ROLE:\n{role}{RESET}")
    # print(f"{BLUE}PROMPT:\n{prompt}{RESET}")
    # print(f"{GREEN}RESPONSE:\n{response_text.choices[0].message.content}{RESET}")
    # print(f"---")
    return response_text.choices[0].message.content