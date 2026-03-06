import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")
client = genai.Client(api_key=api_key)
test_content = client.models.generate_content(model="gemma-3-27b-it", contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.")
output = test_content.text
p_tokens = test_content.usage_metadata.prompt_token_count
r_tokens = test_content.usage_metadata.candidates_token_count
print(output)
print(f"Prompt tokens: {p_tokens}\nResponse tokens: {r_tokens}")
