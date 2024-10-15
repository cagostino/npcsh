import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def generate_text(model_name, prompt, max_length=100):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create a text generation pipeline
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Generate text
    result = generator(prompt, max_length=max_length, num_return_sequences=1)

    return result[0]['generated_text']

prompt = 'Once upon a time'
model_name = 'state-spaces/mamba-2.8b'
example = generate_text(model_name, prompt)
