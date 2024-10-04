# npcsh

Welcome to npcsh, the shell for interacting with NPCs (LLM-powered AI agents). npcsh is meant to be a drop-in replacement shell for any kind of bash/zsh/powershell and allows the user to directly operate their machine through the use of the LLM-powered shell.

Additionally, npcsh introduces a new paradigm of programming for LLMs: npcsh allows users to set up NPC profiles (a la npc_profile.npc) where a user sets the primary directive of the NPC, the tools they want the NPC to use, and other properties of the NPC. NPCs can interact with each other and their primary directives and properties make these relationships explicit through jinja references.

## Dependencies
- ollama
- python >3.10

The default model is currently phi3. 
Download it by running
```
ollama run phi3
```

The user can change the model by setting the environment variable `NPCSH_MODEL` to the desired model name and to change the provider by setting the environment variable `NPCSH_PROVIDER` to the desired provider name.

The provider must be one of ['ollama', 'openai', 'anthropic'] and the model must be one available from those providers.


## Linux install
```bash
sudo apt-get install espeak

sudo apt-get install portaudio19-dev python3-pyaudio

sudo apt-get install alsa-base alsa-utils

sudo apt-get install libcairo2-dev

sudo apt-get install libgirepository1.0-dev


```


pip install npcsh

##Mac install
brew install portaudio
brew install ffmpeg
brew install ollama

brew services start ollama
brew install pygobject3
pip install npcsh


## compilation

Each NPC can be compiled to accomplish their primary directive and then any issues faced will be recorded and associated with the NPC so that it can reference it later through vector search. In any of the modes where a user requests input from an NPC, the NPC will include RAG search results before carrying out the request.


## Base npcsh


In the base npcsh shell, inputs are processed by an LLM. The LLM first determines what kind of a request the user is making and decides which of the available tools or modes will best enable it to accomplish the request. 


### spool mode

Spool mode allows the users to have threaded conversations in the shell, i.e. conversations where context is retained over the course of several turns.
Users can speak with specific NPCs in spool mode by doing ```/spool <npc_name>``` and can exit spool mode by doing ```/exit```.

## Built-in NPCs
Built-in NPCs are NPCs that should offer broad utility to the user and allow them to create more complicated NPCs. These built-in NPCs facilitate the carrying out of many common data processing tasks as well as the ability to run commands and to execute and test programs.

### Bash NPC
The bash NPC is an NPC focused on running bash commands and scripts. The bash NPC can be used to run bash commands and the user can converse with the bash NPC by doing ```/spool bash``` to interrogate it about the commands it has run and the output it has produced.
A user can enter bash mode by typing ```/bash``` and can exit bash mode by typing ```/bq```.
Use the Bash NPC in the profiles of other NPCs by referencing it like ```{{bash}}```.

### Command NPC

The LLM or specific NPC will take the user's request and try to write a command or a script to accomplish the task and then attempt to run it and to tweak it until it works or it's exceeded the number of retries (default=5).

Use the Command NPC by typing ```/cmd <command>```. Chat with the Command NPC in spool mode by typing ```/spool cmd```.
Use the Command NPC in the profiles of other NPCs by referencing it  like ```{{cmd}}```.

### Data NPC

Users can create schemas for recording observations and for exploring and analyzing data.

The Data NPC will asily facilitate the recording of data for individuals in essentially any realm (e.g. recipe testing, one's own blood pressure or  weight, books read, movies watched, daily mood, etc.) without needing to use a tangled web of applications to do so. Observations can be referenced by the generic npcsh LLM shell or by specific NPCs.
Use the Observation NPC by typing ```/data <observation>```.
Chat with the Observation NPC in spool mode by typing ```/spool obs```.
Use the Observation NPC in the profiles of other NPCs by referencing it like ```{{obs}}```. Exit by typing ```/dq```.


### Question NPC

The user can submit a 1-shot question to a general LLM or to a specific NPC.
Use it like
```/question <question> <npc_name>```
or
```/question <question>```

You can also chat with the Question NPC in spool mode by typing ```/spool question```.



### thought mode

This will be like a way to write out some general thoughts to get some 1-shot feedback from a general LLM or a specific NPC.

Use it like
```/thought <thought> <npc_name>```
or
```/thought <thought>```

 
You can also chat with the Thought NPC in spool mode by typing ```/spool thought```.
