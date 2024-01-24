This is a very simple demo of a speech-to-text and then text-to-speech AI.

At this stage there is no async, so there is an unacceptable pause between sentences.

To get this running, follow the below steps:
1. Clone this repo: `git clone https://github.com/nydasco/sam-ai.git`
2. Within the folder, create a virtual environment: `python3 -m venv .venv`
3. Start the environment: `source .venv/bin/activate`
4. Install dependencies: `pip3 install -r requirements.txt`
5. Download the LLM (5.13G): `huggingface-cli download TheBloke/OpenHermes-2.5-Mistral-7B-GGUF openhermes-2.5-mistral-7b.Q5_K_M.gguf --local-dir . --local-dir-use-symlinks False`
6. Run the file: `python3 sam.py`

Note that the first time it runs it will take a while to get started. There are additional models that need to be downloaded. They're smaller than the main LLM.