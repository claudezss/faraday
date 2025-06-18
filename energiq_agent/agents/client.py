from openai import OpenAI


def get_openai_client(
    api_key: str = "EMPTY",
    base_url: str = "http://localhost:11434/v1/",
):
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    return client
