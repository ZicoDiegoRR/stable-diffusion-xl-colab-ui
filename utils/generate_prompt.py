from transformers import pipeline as pipe, set_seed
import random
import re

# Prompt wildcard function
def generate(starting_text, ideas_line, gpt2_pipe):
    if gpt2_pipe:
        # Set seed
        random_seed_for_gpt2 = random.randint(100, 1000000)
        set_seed(random_seed_for_gpt2)

        # Handle empty prompt (generating a prompt based on the ideas.txt)
        if starting_text == "":
            starting_text: str = ideas_line[random.randrange(0, len(ideas_line))].replace("\n", "").lower().capitalize()
            starting_text: str = re.sub(r"[,:\-–.!;?_]", '', starting_text)

        # Generate
        response = gpt2_pipe(starting_text, max_length=(len(starting_text) + random.randint(60, 90)), num_return_sequences=1)
        response_list = []
        for x in response:
            resp = x['generated_text'].strip()
            if resp != starting_text and len(resp) > (len(starting_text) + 4) and resp.endswith((":", "-", "—")) is False:
                response_list.append(resp+'\n')

        # Clean the generated prompt
        response_end = "\n".join(response_list)
        response_end = re.sub('[^ ]+\.[^ ]+','', response_end)
        response_end = response_end.replace("<", "").replace(">", "")

        # Return
        if response_end != "":
            return response_end
        else:
            print("Failed to generate prompt.")
            return starting_text
    else:
        print("Failed to generate prompt.")
        return starting_text
