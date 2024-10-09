# Deep Reinforcement Learning 
## Lecture by Prof. Malte Schilling at University of Muenster

# Submodule usage 

https://git-scm.com/book/en/v2/Git-Tools-Submodules

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



