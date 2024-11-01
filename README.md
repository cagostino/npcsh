<p align="center">
  <img src="npcsh.png" alt="npcsh logo with sibiji the spider">
</p>                                      


# npcsh


Welcome to npcsh, the shell for interacting with NPCs (LLM-powered AI agents) and for coordinating actions and information between the NPCs. 

npcsh is meant to be a drop-in replacement shell for any kind of bash/zsh/powershell and allows the user to directly operate their machine through the use of the LLM-powered shell.

npcsh introduces a new paradigm of programming for LLMs: npcsh allows users to set up NPC profiles (a la npc_profile.npc) where a user sets the primary directive of the NPC, the tools they want the NPC to use, and other properties of the NPC. NPCs can interact with each other and their primary directives and properties make these relationships explicit through jinja references.

With npcsh, we can more seamlessly stick together complex workflows and data processing tasks to form NPC Assembly Lines where pieces of information are evaluated in a sequence by different NPCs and the results are passed along to the next NPC in the sequence. 


## Dependencies
- ollama
- python >3.10

The default model is currently phi3. 
Download it by running
```
ollama run phi3
```

We support inference as well via openai and anthropic. To use them, set an ".env" file up in the folder where you are working and set the API keys there. 

Eventually, we will add the ability to use any huggingface model.

The user can change the default model by setting the environment variable `NPCSH_MODEL` in their ~/.npcshrc to the desired model name and to change the provider by setting the environment variable `NPCSH_PROVIDER` to the desired provider name.

The provider must be one of ['ollama', 'openai', 'anthropic'] and the model must be one available from those providers.



## Linux install
```bash
sudo apt-get install espeak
sudo apt-get install portaudio19-dev python3-pyaudio
sudo apt-get install alsa-base alsa-utils
sudo apt-get install libcairo2-dev
sudo apt-get install libgirepository1.0-dev
sudo apt-get install ffmpeg
pip install npcsh

```



## Mac install
```bash
brew install portaudio
brew install ffmpeg
brew install ollama
brew services start ollama
brew install pygobject3
pip install npcsh
```

## Usage
After it has been pip installed, npcsh can be used as a command line tool. Start it by typing:
```bash
npcsh
```

Once in the npcsh, you can use bash commands or write natural language queries or commands. You can also switch between different modes defined below and you can compile a network of NPCs or use the macro tools we have developed.



## compilation

Each NPC can be compiled to accomplish their primary directive and then any issues faced will be recorded and associated with the NPC so that it can reference it later through vector search. In any of the modes where a user requests input from an NPC, the NPC will include RAG search results before carrying out the request.


## Base npcsh


In the base npcsh shell, inputs are processed by an LLM. The LLM first determines what kind of a request the user is making and decides which of the available tools or modes will best enable it to accomplish the request. 




## Built-in NPCs
Built-in NPCs are NPCs that should offer broad utility to the user and allow them to create more complicated NPCs. These built-in NPCs facilitate the carrying out of many common data processing tasks as well as the ability to run commands and to execute and test programs.



## Other useful tools
### whisper mode
type
```npcsh
/whisper
```
to enter into a voice control mode. It will calibrate for silence so that it will process your input once youve finished speaking and then will tts the response from the  llm.

### spool mode

Spool mode allows the users to have threaded conversations in the shell, i.e. conversations where context is retained over the course of several turns.
Users can speak with specific NPCs in spool mode by doing 
```npcsh
/spool <npc_name>
```
 and can exit spool mode by doing
```npcsh
/exit
```

### Commands

The LLM or specific NPC will take the user's request and try to write a command or a script to accomplish the task and then attempt to run it and to tweak it until it works or it's exceeded the number of retries (default=5).

Use the Command NPC by typing ```/cmd <command>```. Chat with the Command NPC in spool mode by typing ```/spool cmd```.
Use the Command NPC in the profiles of other NPCs by referencing it  like ```{{cmd}}```.


### Question NPC

The user can submit a 1-shot question to a general LLM or to a specific NPC.
Use it like
```npcsh> /sample <question> <npc_name>```
or
```npcsh> /sample <question>```

### Over-the-shoulder 

Over the shoulder allows the user to select an area of the screen and the area will be passed to a vision LLM and then the user can inquire about the image or ask for help with it.
Use it by typing
```npcsh
npcsh> /ots
```
It will pop up with your desktop's native screenshotting capabilities, and allow you to select an area. That area will be saved to ~/.npcsh/screenshots/ and then you will be prompted to pass a question about the image. 
You can also use it on existing files/images by typing 
```npcsh
npcsh> /ots filename
```
and it will also prompt in the same way. 

### data 
Data mode makes it easy to investigate data and ingest it into a local database for later use and inspection.  
begin data mode by typing 
```npcsh
npcsh> /data
```
then load data from a file like
```npcsh
data> load from filename as table_name
```
If it's a tabular file like a csv, you can then perform sql and pandas like operations on the table_name.




