# PendulumModel
Pendulum model simulation for MTH3310 Assignment. 

Clone the repository. GitHub has removed using passwords to clone over https, so you'll have to create a personal access token, following the guide in [this](https://docs.github.com/en/github/authenticating-to-github/keeping-your-account-and-data-secure/creating-a-personal-access-token) link. Then use this PAT when prompted for a password in the following commands. 
```
git clone https://github.com/acol0005/PendulumModel.git
cd PendulumModel
```


Create and activate a new virtual environment. 
```
python3 -m venv pendulumenv
. pendulumenv/bin/activate
```
Ensure that each line in your terminal is now prefixed with (pendulumenv), indicating the virtual environment is active. 
Every time that you open a terminal, check if the '(pendulumenv)' prefix is there. If it isn't, run the `.  pendulumenv/bin/activate` command from the PendulumModel directory to reactivate the virtual environment. 

Install the pre-requisites:
`pip3 install -r requirements.txt`

Running the code:
`python3 main.py`
