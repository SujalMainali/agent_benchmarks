"""LoCoMo-specific prompts and instructions."""

#: Answer-format requirements the LoCoMo scorer depends on (token-overlap F1
#: over short gold answers). Architecture-neutral: it constrains only the
#: *shape* of the answer, never how the agent's input is structured, so it is
#: valid appended onto a full-context, RAG, or <IS>-state agent's own prompt.
#: Delivered via ``RuntimeSpec.format_contract`` for merge-mode drivers.
LOCOMO_FORMAT_CONTRACT = """\
- Reply with ONLY the shortest direct answer: a name, place, number, date,
  or short phrase. No full sentences, no explanations, no reasoning steps,
  no punctuation beyond what the answer itself requires.
- Answer with exactly what the question asks for; do not add dates, names,
  or context that was not asked about.
- If the question asks WHEN something happened, give a specific absolute
  date (or the most precise time you can determine), not a relative phrase
  like "last week".
- If you cannot determine the answer from the information available to you,
  reply exactly: "Information not found."\
"""

LOCOMO_SYSTEM_PROMPT = """
You are a research helper agent specialized in answering questions based on long conversation histories.

Your task is to answer factual questions that refer to information from earlier in the conversation.

Guidelines:
- Be concise and accurate.
- When answering, cite the part of the conversation context that supports your answer.
- If the answer is not in the conversation history, say "Information not found in context" rather than guessing.
- Use the bound tools (web search, document search, note lookup) only if the answer is not already in the provided context.
- Distinguish between information sources:
  - CONTEXT: Information from the conversation history provided.
  - WEB: Information from web search.
  - DOCUMENTS: Information from local project documents.
- Answer in the same language as the question.
"""

LOCOMO_QA_PROMPT = """
Based on the conversation history provided, answer the following question:

{question}

Important:
1. Look for the answer in the provided conversation context first.
2. If you find the answer in the context, cite it explicitly.
3. Only use external tools if the answer is not in the conversation history.
4. Be factual and concise.
5. If you cannot find the answer, say so clearly.
"""

LOCOMO_EVIDENCE_AWARE_PROMPT = """
Based on the conversation history and the provided evidence, answer the following question:

{question}

Evidence provided:
{evidence}

Instructions:
1. First check if the answer is directly in the conversation history.
2. Then check the provided evidence.
3. Combine information from both sources if needed.
4. Always cite which part of the history or evidence supports your answer.
5. Be accurate and concise.
"""

LOCOMO_STRICT_FORMAT_PROMPT = """
Answer the following question in exactly one sentence:

{question}

Conversation context:
{context}

Requirements:
- One sentence only.
- Complete answer in that sentence.
- No follow-up questions or clarifications.
- If you cannot answer, say "Answer not found in context".
"""
