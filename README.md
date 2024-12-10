# EdgeAvatar 

Focussed on the Formar President Franklin D Roosevelt. This project helps us interact with FDR by asking questions and getting answers from him.  

Used Coqui-AI Text to Speech to clone the voice and generate the Audio from Text. Used Open-Ai's wishper model to convert the speech to text and open-Ai's GPT 4o API to generate the answers.   

## Usage 
Make sure system has python>3.7. Tested with python3.11  

create a virtual environment to manage packages
```sh
python3.11 -m venv env
source env/bin/activate
```

Install the dependecies 
```sh
pip install -r requirements.txt
```

Run the code 
```sh
python3.11 main.py
```

