"""One shared prompt format for SFT, distillation, and baseline inference."""

import re


SYSTEM_PROMPT = (
    "You write SQLite queries to answer a user's question. "
    "Use only the supplied database schema and evidence. "
    "Return exactly one read-only SELECT query (WITH is allowed). "
    "Return SQL only, without explanations or markdown."
)


def format_messages(record, feedback=None):
    """Never expose gold_sql in the model input, including repair turns."""
    content = "Database schema:\n" + record["schema"]
    evidence = record.get("evidence", "").strip()
    if evidence:
        content += "\n\nEvidence supplied with the question:\n" + evidence
    content += "\n\nQuestion:\n" + record["question"]
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}]
    for step in feedback or []:
        messages.append({"role": "assistant", "content": step["sql"]})
        messages.append({"role": "user", "content":
                         "SQLite execution failed with this error:\n" + str(step["error"]) +
                         "\nReturn a corrected read-only SQL query. Do not use other tables."})
    return messages


def extract_sql(output):
    """Extract SQL without repairing, inventing, or substituting model outputs."""
    text = output.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    fenced = re.search(r"```(?:sql|sqlite)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    return text
