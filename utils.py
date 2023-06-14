import openai
import tiktoken
from typing import List, Union


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count the number of tokens in a piece of text.
    Encoding cl100k_base is used by gpt-4, gpt-3.5-turbo and text-embedding-ada-002.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def complete_prompt(
    prompt: str, system_role: str, model: str = "gpt-3.5-turbo", temperature: int = 0
) -> str:
    """
    Given a prompt, return a predicted completion.

    system_role: Description of a specific role that we want to set up for the model.
    """
    response = openai.ChatCompletion.create(
        messages=[
            {
                "role": "system",
                "content": system_role,
            },
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=temperature,
    )
    return response["choices"][0]["message"]["content"]


def embed(text: Union[List[str], str], embedding_model: str = "text-embedding-ada-002"):
    """Embed one or multiple pieces of text."""
    return openai.Embedding.create(model=embedding_model, input=text)
