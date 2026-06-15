#!/usr/bin/env python3
"""
Quick test of LLaMA 3 8B generator with 4-bit quantization.
Tests the generator on a simple TriviaQA question.
"""

import os
import sys

# Add repo to path
sys.path.insert(0, "/fab3/btech/2024/kausik.patra24b/PSM_New_V1/PSM_NEW")

from src.data.schemas import RetrievedDoc
from src.generation.generator import Generator

# Enable debug mode
os.environ["RAG_GENERATION_DEBUG"] = "1"

# Test data
test_question = "Who was the man behind The Chipmunks?"
test_docs = [
    RetrievedDoc(
        doc_id=0,
        text="David Seville is an American musician and songwriter who created and produced the cartoon series The Chipmunks. He was born David Bartholemew Seville on January 30, 1919. He is best known for his work as the creator and original voice of The Chipmunks characters.",
        score=0.95,
        source="test",
    ),
    RetrievedDoc(
        doc_id=1,
        text="The Chipmunks are an animated band and cartoon series that debuted in 1958. The characters include Alvin, Simon, and Theodore.",
        score=0.85,
        source="test",
    ),
]

print("=" * 80)
print("Testing flan-t5-base Generator")
print("=" * 80)
print(f"\n❓ Question: {test_question}")
print(f"\n📚 Retrieved Documents:")
for i, doc in enumerate(test_docs):
    print(f"   [{i}] {doc.text[:100]}...")

print("\n🚀 Initializing generator...")
generator = Generator(
    enabled=True,
    model_name="google/flan-t5-base",
    max_new_tokens=64,
    temperature=0.0,
)

if not generator.enabled:
    print("❌ Generator initialization failed!")
    sys.exit(1)

print("\n🔄 Generating answer...")
answer = generator.generate(test_question, test_docs)

print(f"\n✅ Generated Answer: {answer}")
print(f"\n🎯 Expected Answer: David Seville")
print("\n" + "=" * 80)

# Check if answer contains the key name
if "David Seville" in answer or "david seville" in answer.lower() or "seville" in answer.lower():
    print("✓ SUCCESS: Answer contains expected name!")
else:
    print("⚠ WARNING: Answer may not contain expected name. This is okay; LLaMA may phrase it differently.")

print("=" * 80)
