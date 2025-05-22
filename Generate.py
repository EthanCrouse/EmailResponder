import textwrap
import torch

class Generate():
    def __init__(self, model, tokenizer, pipe):
        self.model = model
        self.tokenizer = tokenizer
        self.pipe = pipe
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_prose(self):
        self.model.eval()

        user_text = self.read_data("data/user_text.txt")
        if not user_text:
            raise ValueError("User text is empty. Please provide valid input.")
        
        prompt = f"Read the text and write a few sentences describing the tone:\n\n{user_text}\n\nResponse:"

        full = self.pipe(prompt, max_new_tokens=50)[0]["generated_text"]
        response = full[len(prompt):].strip()
        return response

    def generate_response(self, prose, input, feedback):
        self.model.eval()

        prompt = textwrap.dedent(f"""
        You are writing an email reply. Match the writing style of the following text:

        "{prose}"

        The user wants to reply with the following tone or intent:
        "{feedback}"

        Here is the original email to respond to:
        "{input}"

        Write the full email reply starting below:

        Email:
        """)

        full = self.pipe(prompt, max_new_tokens=50)[0]["generated_text"]
        response = full[len(prompt):].strip()
        return response

    def read_data(self, file_path):
        with open(file_path, 'r') as file:
            return file.read().strip()
