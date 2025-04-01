from npcsh.llm_funcs import stream_with_interrupts

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Ask questions naturally when you need more information.",
    },
    {"role": "user", "content": "What's the weather like there?"},
]

# Run the streaming conversation and print the output
for response in stream_with_interrupts(
    messages=messages, model="gpt-4o-mini", provider="openai"
):
    print(response, end="", flush=True)
