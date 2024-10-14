import gradio as gr
# import langchain
#from google.cloud import aiplatform
import tempfile
import os
import json

from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
import html2text
from PyPDF2 import PdfReader
import tempfile

# Initialize Vertex AI
#aiplatform.init(project='your-gcp-project-id', location='your-region')

ENV_VAR = "GOOGLE_APPLICATION_CREDENTIALS_JSON"

def get_env_var():
	service_account_key_str = os.getenv(ENV_VAR)
	service_account_key = json.loads(service_account_key_str)
	with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
		json.dump(service_account_key, temp_file)
		temp_file.flush()
		os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file.name


# Agent necessary for calling VertexAI using langchain
def get_agent(model="gemini-1.5-flash-001", temperature=0.25):
	agent = ChatVertexAI(model=model, temperature=temperature, max_retries=1)
	return agent

# Function to parse the cv in pdf format
def parse_pdf(pdf_file):
	loader = PdfReader(pdf_file)
	fulltext = ""
	for page in loader.pages:
		fulltext = f"{fulltext}\n{page.extract_text()}"
	return fulltext

# Function to parse job description from URL
def parse_job_description(job_description_link, language='english'):
	try:
		# Fetch the job description from the URL
		response = requests.get(job_description_link)
		
		if response.status_code != 200:
			return f"Failed to fetch job description. Status code: {response.status_code}"
		
		# Parse the HTML content using BeautifulSoup
		soup = BeautifulSoup(response.content, 'html.parser')
		
		# Extract meaningful content, here using <p>, <h1>, <h2>, and <ul>, <li> tags for structure
		job_desc = '\n'.join([str(tag) for tag in soup.find_all(['p', 'h1', 'h2', 'ul', 'li'])])

		# Convert HTML content to Markdown using html2text
		markdown_parser = html2text.HTML2Text()
		markdown_parser.ignore_links = False  # This keeps the links in the markdown
		job_desc_markdown = markdown_parser.handle(job_desc)

		if len(job_desc_markdown.strip()) == 0:
			return "Unable to parse job description. Please edit it manually."
		
		return job_desc_markdown
		
	except Exception as e:
		return f"Error fetching or parsing job description: {str(e)}"

# Function to handle the cover letter generation
def generate_cover_letter(cv_file, parsed_job_desc, language):
	# Extract CV content from PDF (basic implementation)
	if hasattr(cv_file, 'name'):
		cv_path = cv_file.name
	else:
		with tempfile.NamedTemporaryFile(delete=False) as temp:
			temp.write(cv_file.read())
			temp.flush()
			cv_path = temp.name
	
	# Here, you can use a library like PyPDF2 or pdfplumber to extract text
	resume_plain_text = parse_pdf(cv_path)


	cover_letter_template_prompt = PromptTemplate.from_template(
		"""Given the following resume and job listing information, generate a cover letter in {language} as part of the job application. The cover letter should not contain any contact information (to or from) and only contain salutations and a body of at most 3 paragraphs using business causal language. You should highlight any overlap of technology, responsibility or domain present between the job listing and my experience while mentioning why I would be a good fit for the given role. You should use optimistic and affirmative language and end the message with a call to action. Be concise.
		------------
		Resume(Assume that the first few lines are personal details such as name and contact information):
		{resume}
		------------
		Job Listing:
		{job_listing}"""
	)

	cover_letter_prompt = cover_letter_template_prompt.format(language=language, resume=resume_plain_text, job_listing=parsed_job_desc)

	agent = get_agent(temperature=0.7)
	resp = agent.invoke(cover_letter_prompt)
	
	return resp.content

def interface():
	with gr.Blocks() as demo:
		with gr.Row():  # Sidebar area
			parsed_job_desc = None
			with gr.Column(scale=1, min_width=300):  # Sidebar column
				gr.Markdown("### Cover Letter Generator")
				cv_input = gr.File(label="Upload your CV (PDF)", file_types=["pdf"])
				job_desc_link = gr.Textbox(label="Job Description Link", placeholder="Paste job description URL here")
				language = gr.Dropdown(["English", "German"], label="Choose Language", value="English")
				
				parse_button = gr.Button("Parse Job Description")
				# parsed_job_desc = gr.Textbox(label="Parsed Job Description", interactive=True, lines=10)

				# parse_button.click(fn=parse_job_description, inputs=job_desc_link, outputs=parsed_job_desc)

			with gr.Column(scale=2):  # Main content area (Right Column)
				gr.Markdown("### Edit the Parsed Job Description")
				parsed_job_desc = gr.Textbox(label="Parsed Job Description", interactive=True, lines=10)

				generate_button = gr.Button("Generate Cover Letter")
				
				# Connect the left-side parse button to fill the right-side textbox
				parse_button.click(fn=parse_job_description, inputs=job_desc_link, outputs=parsed_job_desc)

		output = gr.Textbox(label="Generated Cover Letter", interactive=False, lines=10)	
		generate_button.click(fn=generate_cover_letter, inputs=[cv_input, parsed_job_desc, language], outputs=output)

	demo.launch()

if __name__ == "__main__":
	get_env_var()
	interface()

