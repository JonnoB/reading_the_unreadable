import json
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type


def truncate_to_n_tokens(text, tokenizer, max_tokens=100):
    # Tokenize the text
    tokens = tokenizer.encode(text, add_special_tokens=False)

    # Truncate to max_tokens
    truncated_tokens = tokens[:max_tokens]

    # Decode back to text
    truncated_text = tokenizer.decode(truncated_tokens)

    return truncated_text


def create_genre_prompt(article_text):
    """
    Generates a prompt for classifying newspaper articles into specific genres.

    Args:
        article_text (str): The text of the article to classify.

    Returns:
        str: A formatted prompt for classifying the article.
    """
    genre_prompt = f"""
    Read the following article.
    :::
    {article_text}
    :::

    You are a machine that classifies newspaper articles. Your response is limited to choices from the following JSON:
        {{
            0: 'news report',
            1: 'editorial',
            2: 'letter',
            3: 'advert',
            4: 'review',
            5: 'poem/song/story',
            6: 'other'
        }}
    You will respond using a single digit in a JSON format.

    For example, given the text "Mr Bronson died today, he was a kind man" your answer would be:
    {{'class': 6}}

    Alternatively, given the text "The prime minister spoke at parliament today" your answer would be:
    {{'class': 0}}
    """
    return genre_prompt


def create_iptc_prompt(article_text):
    """
    Generates a prompt for classifying newspaper articles into specific IPTC categories.

    Args:
        article_text (str): The text of the article to classify.

    Returns:
        str: A formatted prompt for classifying the article.
    """
    iptc_prompt = f"""
    Read the following article.
    :::
    {article_text}
    :::

    You are a machine that classifies newspaper articles. Your response is limited to choices from the following JSON:
        {{
            0: 'arts, culture, entertainment and media',
            1: 'crime, law and justice',
            2: 'disaster, accident and emergency incident',
            3: 'economy, business and finance',
            4: 'education',
            5: 'environment',
            6: 'health',
            7: 'human interest',
            8: 'labour',
            9: 'lifestyle and leisure',
            10: 'politics',
            11: 'religion',
            12: 'science and technology',
            13: 'society',
            14: 'sport',
            15: 'conflict, war and peace',
            16: 'weather',
            17: 'N/A'
        }}
    You will respond using a JSON format.

    For example, given the text "The war with Spain has forced schools to close" your answer would be:
    {{'class': [15, 4]}}

    Alternatively, given the text "The prime minister spoke at parliament today" your answer would be:
    {{'class': [10]}}
    """
    return iptc_prompt


@retry(
    wait=wait_fixed(0.2),  # Wait 0.2 seconds between attempts (5 requests per second)
    stop=stop_after_attempt(3),  # Maximum 3 attempts
    retry=retry_if_exception_type(Exception),
)
def classify_text_with_api(prompt, client, model="mistral-large-latest"):
    try:
        # Get API response
        chat_response = client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

        content = chat_response.choices[0].message.content.strip()

        # Remove any markdown formatting
        content = content.lstrip("```json").rstrip("```").strip()

        def clean_response(text):
            # Replace single quotes with double quotes for JSON compatibility
            text = text.replace("'", '"')
            # Remove any whitespace or newlines
            text = "".join(text.split())
            return text

        # Clean the response
        cleaned_content = clean_response(content)

        # Try multiple parsing methods
        try:
            # Method 1: Direct JSON parsing
            result_dict = json.loads(cleaned_content)
        except json.JSONDecodeError:
            try:
                # Method 2: Python literal eval
                import ast

                result_dict = ast.literal_eval(content)
            except:
                try:
                    # Method 3: Manual parsing for simple cases
                    if "class" in content and ":" in content:
                        # Extract the class value
                        class_str = content.split(":")[1].strip().rstrip("}")
                        if class_str.startswith("[") and class_str.endswith("]"):
                            # Handle list of classes
                            class_values = [
                                int(x.strip())
                                for x in class_str[1:-1].split(",")
                                if x.strip()
                            ]
                            result_dict = {"class": class_values}
                        else:
                            # Handle single class
                            class_value = int(class_str)
                            result_dict = {"class": class_value}
                    else:
                        print(f"Unable to parse response format: {content}")
                        return {"class": 99}
                except:
                    print(f"Failed to parse response: {content}")
                    return {"class": 99}

        # Validate the result
        if not isinstance(result_dict, dict) or "class" not in result_dict:
            print(f"Invalid result format: {result_dict}")
            return {"class": 99}

        return result_dict

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        try:
            print(f"Response content: {content}")
        except:
            print("Could not print response content")
        return {"class": 99}
