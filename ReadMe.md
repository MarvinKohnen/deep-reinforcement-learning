# Deep Reinforcement Learning 
## Lecture by Prof. Malte Schilling at University of Muenster

# Subtree usage 

## First Time Setup 
1. clone this repository
2. invoke the following in root of project:

	`git remote add malte-repo https://zivgitlab.uni-muenster.de/ai-systems/drl_deep_course.git`

	`git remote add janosch-repo https://zivgitlab.uni-muenster.de/jbajorat/drl_intro-python.git`

	`git remote add simon-repo https://zivgitlab.uni-muenster.de/ai-systems/bomberman_rl.git`

## Everyday use

Pull new changes from external repositories:

`git fetch malte-repo`

`git subtree pull --prefix=drl_deep_course malte-repo main --squash`

or respectively: 

`git fetch janosch-repo`

`git subtree pull --prefix=drl_intro_python janosch-repo main --squash`


or respectively:

`git fetch simon-repo`

`git subtree pull --prefix=drl_deep_project simon-repo main --squash`

Proceed to add, commit and push changes as usual.

# Reminder to use virtual environments: 
Invoke the following in root of project

`python3 -m venv <venv_name>`

Activate via `source <venv_name>/bin/activate`

Deactivate via `deactivate`

# Pep8 Style Formatter: Ruff

## Usage 
1. Pip install ruff
2. Pep8 style config file: Ruff.toml Place in root of project. (already in this repo)
3. Install Ruff Extension and Run On Save from Emeraldwalk Extension in VS Code
4. For on save auto format, edit settings.json from VSCode and add the following:

```
"editor.formatOnSave": true,
"[python]": {
	"editor.formatOnSave": true,
	"editor.defaultFormatter": "charliermarsh.ruff"
},
"emeraldwalk.runonsave": {
	"commands": [
		{
			"match": "\\.py$",
			"cmd": "ruff --fix ${file}"
		}
	]
}
```

For further information on pep8 style: https://peps.python.org/pep-0008/



