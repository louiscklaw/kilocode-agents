import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console

import os, sys
import re
from pprint import pprint

from dotenv import load_dotenv

load_dotenv()

helloworld_md = "\n".join(open("./helloworld.md", "r").readlines())

example_Q1_md = "\n".join(open("./example/Q1.md", "r", encoding="utf-8").readlines())
example_A1_md = "\n".join(open("./example/A1.md", "r+", encoding="utf-8").readlines())

# example_Q2_md = "\n".join(open("./example/Q2.md", "r", encoding="utf-8").readlines())
# example_A2_md = "\n".join(open("./example/A2.md", "r+", encoding="utf-8").readlines())

# example_Q3_md = "\n".join(open("./example/Q3.md", "r", encoding="utf-8").readlines())
# example_A3_md = "\n".join(open("./example/A3.md", "r+", encoding="utf-8").readlines())


test_set_Q_md = "\n".join(open("./test_set/Q.md", "r+", encoding="utf-8").readlines())
test_set_A_f = open("./test_set/A.md", "r+", encoding="utf-8")

termination = TextMentionTermination("TERMINATE")


def configure():
    load_dotenv()
    return os.getenv("OPENROUTER_API_KEY")


model_client = OpenAIChatCompletionClient(
    model="deepseek/deepseek-r1-0528:free",
    api_key=configure(),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.1,
    max_retries=3,
    # max_tokens=5000,
    model_info={
        # https://microsoft.github.io/autogen/stable//reference/python/autogen_core.models.html
        "family": "unknown",
        #
        "vision": False,
        "function_calling": False,
        "json_output": True,
        "structured_output": True,
    },
)


async def main() -> None:
    # model_client = OpenAIChatCompletionClient(model="gpt-4o")

    print("initialize bot done")
    assistant = AssistantAgent(
        "content_converter",
        model_client=model_client,
        system_message="Please study the relationship between below Q and A"
        "Q.md"
        "```markdown"
        "<placeholder_for_Q>"
        "```"
        ""
        "A.md"
        "```markdown"
        "<placeholder_for_A>"
        "```"
        ""
        "User will send you a new Q, please convert it to A format."
        "You need to send the content of A.md back to user only and nothing else."
        "You will put `TERMINATE` in your reply when you done. "
        ""
        "".replace("<placeholder_for_Q>", example_Q1_md).replace(
            "<placeholder_for_A>", example_A1_md
        ),
    )

    team = RoundRobinGroupChat(
        [assistant],
        termination_condition=termination,
    )

    print("running team")
    result = await team.run(
        task="Hi, i need your help"
        "Please help to convert the below Q to A format"
        ""
        "Q.md"
        "```markdown"
        "<placeholder_for_Q_md>"
        "```"
        "\n\n"
        "[put-your-answer-here]"
        ""
        "".replace("<placeholder_for_Q_md>", test_set_Q_md)
    )
    console_log = result.messages[-1].content
    console_log = (
        console_log.replace("```markdown", "")
        .replace("```yaml", "")
        .replace("```", "")
        .replace("TERMINATE", "")
        .strip()
    )

    pprint(result.messages)

    test_set_A_f.truncate(0)
    test_set_A_f.writelines(console_log)
    test_set_A_f.close()
    print("done")


asyncio.run(main())
