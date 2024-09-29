# npcsh

Welcome to npcsh, the shell for interacting with NPCs (LLM-powered AI agents). npcsh is meant to be a drop-in replacement shell for any kind of bash/zsh/powershell and allows the user to directly operate their machine through the use of the LLM-powered shell.

Additionally, npcsh introduces a new paradigm of programming for LLMs: npcsh allows users to set up NPC profiles (a la npc_profile.npc) where a user sets the primary directive of the NPC, the tools they want the NPC to use, and other properties of the NPC. NPCs can interact with each other and their primary directives and properties make these relationships explicit through jinja references.

## compilation

Each NPC can be compiled to accomplish their primary directive and then any issues faced will be recorded and associated with the NPC so that it can reference it later through vector search. In any of the modes where a user requests input from an NPC, the NPC will include RAG search results before carrying out the request.



## Modes


### Base npcsh


In the base npcsh shell, inputs are processed by an LLM. The LLM first determines what kind of a request the user is making and decides which of the available tools or modes will best enable it to accomplish the request. 


### Bash mode
A way to enter bash commands without leaving the npcsh environment. The primary benefit for doing this is that each and every input and output in npcsh is recorded along with some environmental context (e.g. current directory). Bash command histories can also be referenced in L


### command mode

The LLM or specific NPC will take the user's request and try to write a bash command to accomplish the task and then attempt to run it and to tweak it until it works or it's exceeded the number of retries (default=5).


### observation mode

Users can create schemas for recording observations. The idea here is to more easily facilitate the recording of data for individuals in essentially any realm (e.g. recipe testing, one's own blood pressure or  weight, books read, movies watched, daily mood, etc.) without needing to use a tangled web of applications to do so. Observations can be referenced by the generic npcsh LLM shell or by specific NPCs.


### question mode

The user can submit a question to a general LLM or to a specific NPC.



### thought mode
This will be like a way to write out some general thoughts to get some feedback from a general LLM or a specific NPC.



### spool mode

Spool mode allows the users to have threaded conversations in the shell, i.e. conversations where context is retained over the course of several turns.


 

