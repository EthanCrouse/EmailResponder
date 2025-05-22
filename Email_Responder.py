from Email_Reader import EmailReader
from Generate import Generate 
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

"""
Email Responder Class
This class is responsible for fetching emails, generating prose style, and creating responses based on the fetched emails.
It uses the Generate class for text generation and the EmailReader class for email handling.
It uses the Hugging Face Transformers library to load a pre-trained model and tokenizer.
"""
class Email_Responder:
    def __init__(self, email_config):
        self.email = email_config

        # Use a pipeline as a high-level helper
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B",
            torch_dtype=torch.float16,               # Hugging Face will shard layers across your GPU(s)
            low_cpu_mem_usage=True,          # minimize CPU RAM during load
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0                         # GPU 0
        )  

        self.generator = Generate(model, tokenizer, pipe)

    def get_email(self):
        email_reader = EmailReader(
            host=self.email["host"],
            user=self.email["user"],
            password=self.email["password"],
            port=self.email["port"]
        )
        email = email_reader.fetch_emails(count=1)
        return email

    def get_prose(self):
        return self.generator.generate_prose()

    def get_response(self, prose, input, feedback):
        return self.generator.generate_response(prose, input, feedback)
    
    def get_feedback(self, message):
        print("Message: ", message)
        feedback = input("How would you like to respond? - ") 
        return feedback
    
    def reset_model(self):
            """
            Resets the language model to its initial state.
            This can be used to clear any cached states or temporary data.
            """
            model_id = "tiiuae/falcon-rw-1b"  # known to work
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)  # or "cpu"
            model.eval()

            # Set pad token to eos token for generation
            tokenizer.pad_token = tokenizer.eos_token  
            self.generator = Generate(model, tokenizer)

    def save_prose_to_file(self, prose: str, directory: str = "/data", filename: str = "prose.txt"):
        """
        Saves the given prose string into a file under `directory/filename`.
        Creates the directory if it doesn't exist.

        :param prose:    The text you want to write.
        :param directory: Path to the folder (default: '../data').
        :param filename:  Name of the file (default: 'prose.txt').
        """
        # Ensure the target directory exists
        os.makedirs(directory, exist_ok=True)

        # Build the full path and write the file
        filepath = os.path.join(directory, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(prose)

    def run(self):

        if not os.path.isfile("/data/prose.txt"):
            # Generate prose style from the model
            prose = self.get_prose()
            self.save_prose_to_file(prose, "/data", "prose.txt")
            print("Generated prose & saved to /data/prose.txt")
            return 0
        
        prose = self.generator.read_data("/data/prose.txt")
        # Fetch the latest email
        email = self.get_email()
        if not email:
            print("No emails to process.")
            return

        feedback = self.get_feedback(email[0]["body"])

        self.reset_model

        # Generate a response based on the prose and the email content
        response = self.get_response(prose, email[0]["body"], feedback)

        # Store the results
        output = "-" * 100 + "\n"
        output += "PROSE:\n"
        output += prose + "\n"
        output += "-" * 100 + "\n"
        output += "INPUT:\n"
        output += email[0]["body"] + "\n"
        output += "-" * 100 + "\n"
        output += "RESPONSE:\n"
        output += response

        return output