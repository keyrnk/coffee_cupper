import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import re

def init_model_for_rag():
    base_model = 'Qwen/Qwen2.5-3B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model).to("mps" if torch.mps.is_available() else "cpu")
    print("model loaded")
    return model, tokenizer

def get_first_three_sentences(text: str) -> str:
    pattern = re.compile(r'Example\s+3', re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return text[:match.start()].strip()

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    first_three = sentences[:3]
    return " ".join(first_three)

def predict_with_prompt(prompt_model, prompt_tokenizer, coffee_map, coffee_labels, coffee_dataset, max_new_tokens=3):
    mask = ((coffee_dataset['country_of_origin'] == coffee_map['country_of_origin']) & (coffee_dataset['roast'] == coffee_map['roast']))
    filtered = coffee_dataset[mask]
    if len(filtered) > 0:
        example = coffee_dataset[mask].iloc[0]['blind_assessment']
    else:
        example = coffee_dataset.iloc[0]['blind_assessment']

    prompt_text = (f'''Write about coffee 3 sentences with following words: {coffee_labels}.
                       Example 1: {example}
                       Example 2:''')
    print(f"prompt text: {prompt_text}")
    prompt_inputs = prompt_tokenizer(prompt_text, return_tensors="pt").to(prompt_model.device)

    with torch.no_grad():
        out = prompt_model.generate(
            **prompt_inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            pad_token_id=prompt_tokenizer.eos_token_id
        )

    generated_text = prompt_tokenizer.decode(out[0][prompt_inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return get_first_three_sentences(generated_text)

def get_prediction_from_probs(probs):
    label_names = ['roasted', 'spices', 'nutty/cocoa', 'sweet', 'floral', 'fruit', 'sour/fermented', 'green/vegetative',
                   'other']
    best_thresholds = [0.1, 0.15, 0.4, 0.3, 0.2, 0.5, 0.05, 0.1, 0.15]
    class_boosts = {6: 5, 7: 2.5, 0: 3.0, 1: 2.0, 2: 1.0, 3: 2, 4: 1.5, 5: 1.0, 8: 2}
    predicted_labels_text = []

    for row_probs in probs:
        passed_classes = []
        for i, p in enumerate(row_probs):
            threshold = best_thresholds[i]
            if p > threshold:
                boosted_score = p * class_boosts.get(i, 1.0)
                passed_classes.append((i, boosted_score, p))

        passed_classes.sort(key=lambda x: x[1], reverse=True)

        selected_classes = []

        if len(passed_classes) > 0:
            selected_classes.append(passed_classes[0])

            for i in range(1, min(len(passed_classes), 4)):
                current_score = passed_classes[i][1]
                prev_score = passed_classes[i - 1][1]

                confidence_ratio = current_score / prev_score

                if confidence_ratio < 0.5:
                    break

                selected_classes.append(passed_classes[i])

        indices = [item[0] for item in selected_classes]
        names = [label_names[i] for i in indices]
        predicted_labels_text.append(names)
    return predicted_labels_text

def predict_coffee_taste(coffee_map, catboost_model, rag_model, rag_tokenizer):

    coffee_dataset = pd.read_csv('datasets/search_df.csv')

    X_columns = ['country_of_origin', 'processing_method', 'roast', 'variety_Anacafe 14',
     'variety_Arabica', 'variety_Arusha', 'variety_Blue Mountain',
     'variety_Bourbon', 'variety_Castillo', 'variety_Catimor',
     'variety_Catuai', 'variety_Catucai', 'variety_Caturra',
     'variety_Ethiopian Heirlooms', 'variety_Geisha', 'variety_Guji',
     'variety_Huila', 'variety_Java', 'variety_Mandheling', 'variety_Moka',
     'variety_Mundo Novo', 'variety_Pacamara', 'variety_Pacas',
     'variety_Peaberry', 'variety_Ruiru 11', 'variety_SL14', 'variety_SL28',
     'variety_SL34', 'variety_Sidamo', 'variety_Sidra', 'variety_Sulawesi',
     'variety_Sumatra', 'variety_Sumatra Lintong', 'variety_Typica',
     'variety_Yellow Bourbon', 'variety_Yirgacheffe']

    row_dict = {col: 0 for col in X_columns}
    row_dict['country_of_origin'] = coffee_map['country_of_origin']
    row_dict['processing_method'] = coffee_map['processing_method']
    row_dict['roast'] = coffee_map['roast']
    for variety in coffee_map['variety']:
        row_dict['variety_' + variety] = 1

    df_row = pd.DataFrame([row_dict])[X_columns]
    df_row.head()
    probs = catboost_model.predict_proba(df_row)

    predictions = get_prediction_from_probs(probs)
    prompt_answer = predict_with_prompt(rag_model, rag_tokenizer, coffee_map, predictions[0], coffee_dataset, 100)
    return predictions, prompt_answer
