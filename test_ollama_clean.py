"""Test script: clean one caption file with Ollama."""
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from data_forge.tools.ollama import OllamaClient

d = Path("d:/EF_Fish/2048")
txts = sorted(d.glob("*.txt"))

# Test with file #3
test_file = txts[2]
original = test_file.read_text(encoding="utf-8")
print(f"Test file: {test_file.name}")
print(f"Original ({len(original)} chars):")
print(original[:300])
print("...\n")

prompt = (
    "You are a text cleaner. Remove ALL art-style and medium descriptions from "
    "the following image caption. Specifically remove:\n"
    '- Phrases like "digital illustration", "watercolor", "painting", "drawing", '
    '"sketch", "rendering", "photograph", "realistic style", "colored pencil"\n'
    "- Any description of HOW the image was created (the medium/technique)\n\n"
    "Keep ONLY the factual description of the subject itself (what the fish looks "
    "like — its colors, shape, fins, scales, features).\n\n"
    "Return ONLY the cleaned text. No explanations, no markdown, no quotes. "
    "Do not add anything new.\n\n"
    "Original text:\n"
    + original
)

print("Sending to qwen3.5:9b...")
client = OllamaClient(timeout=600)
result = client.generate(prompt, model="qwen3.5:9b", temperature=0.1)
print(f"\nResult ({len(result)} chars):")
print(result[:600])
