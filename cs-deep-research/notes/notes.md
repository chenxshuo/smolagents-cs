# Learning Notes 

## CodeAgent 

### build system prompt based on default template + specific variables

```python
prompt_templates = prompt_templates or yaml.safe_load(
            importlib.resources.files("smolagents.prompts").joinpath("code_agent.yaml").read_text()
        )
```
- path: `src/smolagents/prompts/code_agent.yaml` 

Reading the default system prompt from this path yaml, and populate with specific variables:

```python
system_prompt = populate_template(
    self.prompt_templates["system_prompt"],
    variables={
        "tools": self.tools,
        "managed_agents": self.managed_agents,
        "authorized_imports": (
            "You can import from any package you want."
            if "*" in self.authorized_imports
            else str(self.authorized_imports)
        ),
    },
)
```

**system prompt + variables**

1. general prompt
2. examples 
3. specify variables
    - specify tools 
    - specify managed agents 
4. declare rules such as formats, imports, etc.

### `step` method is the main method of the agent subclass 

- perform one step in ReAct Framework: thinks + acts + observe the result 


1. write memory to messages (method from `MultiStepAgent`)
  - system prompt
  - memory steps 
2. call model with messages to get out model output 
3. parse model output to get code action 
4. call tools 

