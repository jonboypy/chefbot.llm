# ChefBot.llm

**Description:** LLM based kitchen assistant that scrapes & summarizes recipe articles (e.g. from Pinterest).

## Version 1

1. Use large LLM (llama3-8B-Instruct) to curate an instruct finetuning dataset.
    - Develop some sort of dataset generator class that handles everything
    - will need to manually curate recipe article URLs to use
    - in version 1, just provide a bulleted list of the recipe ingredients and instructions.

2. finetune smaller LLM (tinyllama?) on generated dataset (via LoRA / qLoRA?)

3. Deploy finetuned small LLM to iOS
    - INT4?
    - leverage Apple Neural Engine (ANE)

4. Develop frontend iOS app with Swift
    - User either copy/pastes URL into app 
    or 
    - User can open Pinterest article in app from Pinterest and automatically run LLM on it

### Key information to gather during Version 1

- Can large LLM reliably create an accurate instruct finetuning dataset?

- what finetuning methods work the best? Do I need qLoRA with small model?

- Can I deploy huggingface model directly? do I need to write custom model/model-components?

- Can iOS handle the model size? (i.e. performance & memory limitations)

- How much of a hurdle is it to target ANEs?

- Can an article be opened directly from Pinterest?
