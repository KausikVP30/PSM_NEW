# from __future__ import annotations

# from typing import List

# from src.data.schemas import RetrievedDoc


# class PromptAssembler:
#     def __init__(self, max_chars: int = 4000) -> None:
#         self.max_chars = max_chars

#     def assemble(self, question: str, docs: List[RetrievedDoc], memory_answer: str | None = None) -> str:
#         blocks = ["You are a QA assistant.", f"Question: {question}"]
#         if memory_answer:
#             blocks.append(f"Prior memory hint: {memory_answer}")

#         for i, doc in enumerate(docs, start=1):
#             blocks.append(f"Doc {i}: {doc.text}")

#         blocks.append("Answer concisely and factually based on evidence.")
#         prompt = "\n".join(blocks)
#         return prompt[: self.max_chars]


from __future__ import annotations

from typing import List

from src.data.schemas import RetrievedDoc


class PromptAssembler:
    def __init__(self, max_chars: int = 4000, max_docs: int = 3) -> None:
        self.max_chars = max_chars
        self.max_docs = max_docs

    def assemble(
        self,
        question: str,
        docs: List[RetrievedDoc],
        memory_answer: str | None = None,
    ) -> str:
        blocks: List[str] = []

        # 🔒 Strong system instruction (forces grounding)
        blocks.append("You are a factual question-answering system.")
        blocks.append("Answer ONLY using the provided context.")
        blocks.append("If the answer is not present in the context, reply with 'unknown'.")
        blocks.append("Do NOT use prior knowledge outside the context.")
        blocks.append("")

        # ❓ Question
        blocks.append(f"Question: {question}")
        blocks.append("")

        # ⚠️ Memory is treated as weak hint (not truth)
        if memory_answer:
            blocks.append(
                f"Possible prior answer (may be incorrect, use only if supported by context): {memory_answer}"
            )
            blocks.append("")

        # 📚 Context section
        blocks.append("Context:")

        # Limit number of docs to reduce noise
        selected_docs = docs[: self.max_docs]

        for i, doc in enumerate(selected_docs, start=1):
            text = doc.text.strip()
            if not text:
                continue

            # Prevent overly long chunks from dominating
            truncated_text = text[:1000]
            blocks.append(f"{i}. {truncated_text}")

        blocks.append("")

        # 🎯 Final instruction (forces concise grounded answer)
        blocks.append(
            "Answer (concise, factual, and strictly based on the context above):"
        )

        # 🔧 Build prompt safely (avoid cutting instructions)
        prompt = "\n".join(blocks)

        if len(prompt) > self.max_chars:
            # Trim context only, preserve instructions + question
            head = blocks[:6]  # instructions + question
            tail = blocks[-1:]  # final answer instruction

            # Keep as many docs as fit
            context_blocks = []
            current_len = len("\n".join(head + tail))

            for b in blocks[6:-1]:
                if current_len + len(b) < self.max_chars:
                    context_blocks.append(b)
                    current_len += len(b)
                else:
                    break

            prompt = "\n".join(head + context_blocks + tail)

        return prompt
