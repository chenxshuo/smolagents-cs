# -*- coding: utf-8 -*-
""""""

import logging


logger = logging.getLogger(__name__)

PROMPTS = {
    "search_agent_description": """A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like 'Find me this information (...)' rather than a few keywords.
    """
}
