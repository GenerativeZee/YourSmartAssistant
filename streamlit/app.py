import streamlit as st
import requests
import json

st.title("Personal Assistant Bot...")

def convert_to_valid_json(input_data):
    try:
        # First try to parse the input as-is (in case it's already valid)
        parsed = json.loads(input_data) if isinstance(input_data, str) else input_data
        
        # Handle the key_entities if it's a string containing JSON
        if isinstance(parsed.get('key_entities'), str) and '```json' in parsed['key_entities']:
            # Extract the JSON part from between the ```
            json_str = parsed['key_entities'].split('```json')[1].split('```')[0].strip()
            parsed['key_entities'] = json.loads(json_str)
        
        # Handle follow_up_questions if it's a string with bullet points
        if isinstance(parsed.get('follow_up_questions'), str):
            questions = [q.strip('- ').strip() for q in parsed['follow_up_questions'].split('\n') if q.strip()]
            parsed['follow_up_questions'] = questions
        
        # Convert back to JSON string to validate
        json.dumps(parsed)
        return parsed
    except (json.JSONDecodeError, AttributeError, KeyError):
        # If any error occurs, return the original input
        return input_data if not isinstance(input_data, str) else json.loads(input_data)


user_query = st.text_input("Enter your request:")


if st.button("Submit"):
    print("Hellooooooo: ", user_query)
    if user_query.strip() == "":
        st.warning("Please enter a query.")
    else:
        url = "http://127.0.0.1:9090/response"
        payload = {"query": user_query}
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                st.success("API Response:")
                response = convert_to_valid_json(response.json())
                print("valid_json: ", json.dumps(response, indent=2))
                st.json(response) 
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Error connecting to API: {e}")
