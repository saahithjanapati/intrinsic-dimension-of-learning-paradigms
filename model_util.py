import torch


def generate_new_texts(model, tokenizer, texts, max_new_tokens=5):
    """
    Generates new texts by appending generated tokens to the input texts.
    """
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    batch_inputs = tokenizer(texts, return_tensors="pt", padding=True).to('cuda')
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=batch_inputs['input_ids'],
            attention_mask=batch_inputs['attention_mask'],
            max_new_tokens=max_new_tokens
        )
    
    new_texts = []
    for i, input_text in enumerate(texts):
        input_ids = batch_inputs['input_ids'][i]
        generated_ids = outputs[i][len(input_ids):]  # Skip input ids
        new_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        new_texts.append(new_text)
    
    return new_texts



def get_hidden_states_and_logits_for_last_token(prompts, model, tokenizer, verbose=False, padding_side="left"):
    """
    Extracts the hidden states for the last token in each prompt.
    """    
    # Set the padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    
    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract hidden states
    hidden_states = outputs.hidden_states
    
    input_ids = inputs['input_ids']
    batch_size, seq_len = input_ids.shape
    

    last_token_logits = None
    logits = outputs.logits
    
    # Determine the index of the last non-padding token
    if padding_side == "left":
        last_token_indices = torch.ones(batch_size, dtype=torch.int).to(device) * (seq_len - 1)
        last_token_logits = logits[range(batch_size), last_token_indices]

    else:  # right padding
        last_token_indices = (input_ids != tokenizer.pad_token_id).long().sum(dim=1) - 1
        last_token_logits = logits[range(batch_size), last_token_indices]
    
    if verbose:
        print("Verbose mode activated. Printing input details and last token indices.")
        for key, value in inputs.items():
            print(f"{key}: {value}")
        print(f"last_token_indices: {last_token_indices}")
        for x in input_ids:
            print(tokenizer.decode(x))
    
    # Gather the last non-padding token hidden states across all layers
    last_hidden_states = []
    for i in range(batch_size):
        last_idx = last_token_indices[i]
        last_token_hidden_states = [layer[i, last_idx].unsqueeze(0) for layer in hidden_states]
        last_hidden_states.append(torch.cat(last_token_hidden_states, dim=0))
    



    print("Hidden state extraction for last tokens completed.")
    
    del inputs
    del hidden_states
    del outputs
    torch.cuda.empty_cache()
    
    return last_hidden_states, last_token_logits