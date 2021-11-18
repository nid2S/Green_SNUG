from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, GPT2LMHeadModel
from transformers import TFAutoModelForCausalLM, TFGPT2LMHeadModel, GPT2TokenizerFast
import torch, tensorflow

def HF_example(line: int = 5):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

    # Let's chat for 5 lines
    for step in range(line):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')
        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
        # generated a response while limiting the total chat history to 1000 tokens,
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        # pretty print last ouput tokens from bot
        print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))

def HF_korean_example():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PreTrainedTokenizerFast.from_pretrained('byeongal/Ko-DialoGPT')
    model = GPT2LMHeadModel.from_pretrained('byeongal/Ko-DialoGPT').to(device)

    past_user_inputs = []
    generated_responses = []

    while True:
        user_input = input(">> User:")
        if user_input == 'bye':
            break
        text_idx = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        for i in range(len(generated_responses) - 1, len(generated_responses) - 3, -1):
            if i < 0:
                break
            encoded_vector = tokenizer.encode(generated_responses[i] + tokenizer.eos_token, return_tensors='pt')
            if text_idx.shape[-1] + encoded_vector.shape[-1] < 1000:
                text_idx = torch.cat([encoded_vector, text_idx], dim=-1)
            else:
                break
            encoded_vector = tokenizer.encode(past_user_inputs[i] + tokenizer.eos_token, return_tensors='pt')
            if text_idx.shape[-1] + encoded_vector.shape[-1] < 1000:
                text_idx = torch.cat([encoded_vector, text_idx], dim=-1)
            else:
                break
        text_idx = text_idx.to(device)
        inference_output = model.generate(
            text_idx,
            max_length=1000,
            num_beams=5,
            top_k=20,
            no_repeat_ngram_size=4,
            length_penalty=0.65,
            repetition_penalty=2.0,
        )
        inference_output = inference_output.tolist()
        bot_response = tokenizer.decode(inference_output[0][text_idx.shape[-1]:], skip_special_tokens=True)
        print(f"Bot: {bot_response}")
        past_user_inputs.append(user_input)
        generated_responses.append(bot_response)
