Code for the paper [_Speciesism in AI: Evaluating Discrimination Against Animals in Large Language Models_](https://arxiv.org/abs/2508.11534)


## Developer instructions

Use `pipreqs . --force` to update the `requirements.txt` file (`pip install pipreqs`).

## Repository layout
* the main file with the logic, parsing, and exporting is in `main.py`
* `requirements.txt` are the external python packages the code uses
* the `.env.template` file is an example `.env` file. I use a `.env` file to store the API keys, as well as other parameters that are used in the `main.py` file
* the `src/` folder contains some helper methods/types at the top level
* the code in the `src/generators/` folder is not used from what I remember
* the files in `src/models/` are interfaces for interacting with various LLM APIs.

There are then two important folders for running the experiments: `prompts/` and `responses/`.
The `prompts/` folder contains a subfolder for each experiment type/survey (some surveys were unused). Within each prompt subfolder contains three files:

1. `context.txt`: Gives the instructions for the survey, at the system level if possible
2. `formatting_context.txt`: Gives specific instructions on how to format responses (in JSON).
3. `questions.txt`: JSON formatted version of the questions from the survey

The `responses/` folder contains the responses of each of the LLMs for each of the surveys (some surveys were not used). I generated many responses and only kept the first 50 that were formatted correctly. Only the csv files that have exactly 50 rows are actually used: these are the "master" data files. Each numbered column corresponds to an `"id"` in the corresponding `prompts/**/questions.txt` file.

The `notebooks/` folder contains jupyter notebooks of all of the code I used to create the figures and the tables.

The `plots/` folder contains exported plots from the jupyter notebooks.

All other files/folders not mentioned are not used.
