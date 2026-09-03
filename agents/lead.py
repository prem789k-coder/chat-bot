from agents.base import FieldCollector
from core.llm import chat_llm
from models import Lead
from schemas.state import LeadFields


class Prompts:
    EXTRACT = """Extract sales-lead details from the full conversation below. Only fill a \
field if the user has actually stated it - leave it null if not mentioned, do not guess. \
For company_size, give a single integer - if the user gave a range, use the upper bound.

Conversation:
{history}"""

    FIELD_QUESTIONS = {
        "company_size": "roughly how many people would be using it",
        "current_tool": "what they're currently using to manage this",
        "use_case": "the main thing they're hoping to solve",
        "timeline": "their timeline for switching",
        "email": "the best email to reach them at",
    }


class LeadAgent(FieldCollector):
    field_questions = Prompts.FIELD_QUESTIONS
    destination = "pass this along to our sales team"
    fields_cls = LeadFields
    model_cls = Lead
    llm = chat_llm
    extract_prompt = Prompts.EXTRACT


_agent = LeadAgent()
process = _agent.process