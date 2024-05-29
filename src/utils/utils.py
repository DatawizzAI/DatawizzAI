import json


def try_parse_json(sample_output):
    """
    Try to parse a string as JSON.

    Parameters:
    - sample_output (str): The string to parse.

    Returns:
    dict or None: The parsed JSON dictionary, or None if parsing fails.
    """
    try:
        parsed_json = json.loads(sample_output)
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None


def parse_output(sample_output, results_dict=None):
    """
    Parse the output of data generation.

    Parameters:
    - sample_output (str): The output string to parse.
    - results_dict (dict, optional): The dictionary to store the parsed data. If None, a new dictionary will be created.

    Returns:
    dict or None: The updated results dictionary containing parsed data, or None if an error occurs during parsing.
    """
    if results_dict is None:
        results_dict = {}

    try:
        parsed_dict = try_parse_json(sample_output)
        if parsed_dict is not None:
            for key, value in parsed_dict.items():
                if key not in results_dict:
                    results_dict[key] = pd.DataFrame(pd.json_normalize(value))
                else:
                    results_dict[key] = pd.concat([results_dict[key], pd.DataFrame(pd.json_normalize(value))],
                                                  ignore_index=True)

        return results_dict
    except Exception as ex:
        print(f"Error during parsing: {ex}")
        return None


def parse_content_generated_concurrently(content, max_trials=3):
    """
    Parse and aggregate content generated concurrently.

    Parameters:
    - content (List[str]): List of responses from concurrent data generation tasks.
    - max_trials (int, optional): Maximum number of trials to attempt parsing for each task. Defaults to 3.

    Returns:
    dict: Aggregated and parsed content.
    """
    output = {}

    for task in content:
        trial = 0
        while trial < max_trials:
            try:
                task_res = try_parse_json(task)
                for key, value in task_res.items():
                    if key not in output:
                        output[key] = pd.DataFrame(pd.json_normalize(value))
                    else:
                        output[key] = pd.concat([output[key], pd.DataFrame(pd.json_normalize(value))],
                                                ignore_index=True)
                break  # Break out of the trial loop if successful parsing
            except Exception as ex:
                print(f"Error during parsing (trial {trial + 1}): {ex}")
                trial += 1

        if trial == max_trials:
            print(f"Reached maximum trials for parsing. Skipping task result.")

    return output


def dataframe_to_json(input_df, key_name):
    """
    Convert a DataFrame to a JSON string.

    Parameters:
    - input_df (DataFrame): The DataFrame to convert.
    - key_name (str): The key name for the JSON object.

    Returns:
    str: The JSON string representation of the DataFrame.
    """
    json_sample = input_df.to_json(orient='records')
    results = f'{{"{key_name}": {json_sample}}}'
    return results


import os
import pandas as pd


def create_json_sample_from_csv(file_path, file_name, num_samples=5):
    """
    Create a JSON sample from a CSV file.

    Parameters:
    - file_path (str): The path to the directory containing the CSV file.
    - file_name (str): The name of the CSV file.
    - num_samples (int, optional): The number of samples to include in the JSON. Defaults to 5.

    Returns:
    str: A JSON string containing the sampled data.
    """
    # Construct the full file path
    full_file_path = os.path.join(file_path, file_name)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(full_file_path)

    # Sample the DataFrame
    df_sample = df.sample(n=num_samples)

    # Convert the sampled DataFrame to JSON
    json_sample = df_sample.to_json(orient='records')

    # Combine the JSON sample with the file name
    json_result = "{" + f"\"{file_name}\": {json_sample}" + "}"

    return json_result


def create_json_sample_from_dataframes_dictionary(dataframes_dictionary, num_samples=5):
    """
    Create a JSON sample from a CSV file.

    Parameters:
    - dataframes_dictionary (a dictionary): A dictionary with pairs of data name (string) and Dataframe object, that holds the input tabular data.
    - num_samples (int, optional): The number of samples to include in the JSON. Defaults to 5.

    Returns:
    str: A JSON string containing the sampled data.
    """

    json_result = ''

    # Iterate over each dataframe
    for df_name, df in dataframes_dictionary.items():
        # Sample the DataFrame
        df_sample = df.sample(n=num_samples)

        # Convert the sampled DataFrame to JSON
        json_sample = df_sample.to_json(orient='records')

        # Combine the JSON sample with the file name
        json_result = f"\"{df_name}\": {json_sample}"

    # Add {} around
    json_result = "{" + json_result + "}"

    return json_result


def compose_query_message(query, region=None, language=None, task_specifications=None):
    """
    Compose the query message based on region and language parameters.

    Parameters:
    - query (str): The description of the required content.
    - region (str): The required region.
    - language (str): The required language.

    Returns:
    str: The composed query message.
    """
    query_msg = f"Data description: {query}"
    if region:
        query_msg += f" ; The required region: {region}"
    if language:
        query_msg += f" ; All texts should be translated to {language} language."
    if task_specifications:
        query_msg += f" ; The guidance given by an expert: {task_specifications}."
    return query_msg
