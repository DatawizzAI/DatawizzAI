import json

from langchain.prompts import PromptTemplate
from src.utils.utils import *
import random
import math
import asyncio


class DataAugmentor:
    def __init__(self, llm, structure='', batch_size=10, verbose=True):
        """                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        Initialize the DataAugmentor with the specified parameters.

        Parameters:
        - llm: The language model used for data generation.
        - structure: The structure of the data to be generated.
        """
        self.llm = llm
        self.structure = structure
        self.batch_size = batch_size
        self.verbose = verbose
        self.examples_data = None
        self.leading_key = None
        self.task_specifications = None
        self.previous_generated_batch = 'unknown'

        data_augmentor_template = (
            f"You are a system that specializes in generating synthetic data according to user requests."
            f"Generate {batch_size} Sample synthetic items(records in a table or the equivalent elements in unstructured text) "
            f"for the data described below. Follow the required structure (see below), and generate different values."
            f"For the following task: {{human_input}}."
            f"Required Structure: {{structure}}."
            f"The previous generated batch (maintain sequential patterns if needed): {{previous_generated_batch}}."
            f"Add Randomness: {{randomness}}."
            ###f"It is critical that you don't repeat the same values every time you get this prompt."
            f"Ensure that the generated texts are distinct and varied, without repeating any names from previous requests, even across different sessions."
            f"\nGenerated Output:"
        )

        data_augmentor_prompt = PromptTemplate(
            input_variables=["human_input","structure","previous_generated_batch", "randomness"], template=data_augmentor_template
        )

        self.data_augmentor_chain = data_augmentor_prompt | llm

    # Set a dataframe to be used as example. (Currently supports only a single table structure)
    def set_examples_dataframe(self, dataframe, data_name):
        """
        Set a dataframe to be used as example data.

        Parameters:
        - dataframe: The pandas DataFrame containing example data.
        - data_name: The key to identify the leading table in the generated data.
        """
        self.examples_data = dataframe
        self.leading_key = data_name

    def preview_output_sample(self, query=None, region=None, language=None,task_specifications=None, num_records=3, leading_key='',
                              max_retries=3,
                              total_max_retries=10):
        """
        Extract sample data based on a user query.

        Parameters:
        - query (str, optional): The description of the required content. Defaults to None.
        - region (str, optional): The required region. Defaults to None.
        - language (str, optional): The required language. Defaults to None.
        - num_records (int): The desired number of synthetic records to generate.
        - leading_key (str): The key to identify the leading table in the generated data.
        - max_retries (int, optional): Maximum number of retries for a single iteration. Defaults to 3.
        - total_max_retries (int, optional): Maximum cumulative number of retries across all iterations. Defaults to 10.

        Returns:
        dict: A dictionary containing the generated synthetic data.
        """
        return self.generate_data(query=query, region=region, language=language, task_specifications=task_specifications, num_records=num_records,
                                leading_key=leading_key, max_retries=max_retries, total_max_retries=total_max_retries)

    def generate_data(self, query=None, region=None, language=None, task_specifications=None, num_records=3, leading_key='', output_format = 2, max_retries=3,
                    total_max_retries=5):
        """
        Generate synthetic data using the configured data generation chain.

        Parameters:
        - query (str, optional): The description of the required content. Defaults to None.
        - region (str, optional): The required region. Defaults to None.
        - language (str, optional): The required language. Defaults to None.
        - num_records (int): The desired number of synthetic records to generate.
        - leading_key (str): The key to identify the leading table in the generated data.
        - max_retries (int, optional): Maximum number of retries for a single iteration. Defaults to 3.
        - total_max_retries (int, optional): Maximum cumulative number of retries across all iterations. Defaults to 10.

        Returns:
        dict: A dictionary containing the generated synthetic data.
        """
        STRING_ = 0
        JSON_ = 1
        DATAFRAME_DICT_ = 2

        query_msg = compose_query_message(query, region, language, task_specifications)

        results_dict = {}
        n = 0
        total_retries = 0

        try:
            while n < num_records:
                retries = 0
                while retries < max_retries:
                    try:
                        if self.examples_data is not None:
                            cur_structure = self.examples_data.sample(n=self.batch_size)
                            cur_structure = dataframe_to_json(cur_structure, self.leading_key)
                        else:
                            cur_structure = self.structure

                        res = self.data_augmentor_chain.invoke({"structure":cur_structure, "randomness":random.random(),
                                                               "human_input":query_msg,"previous_generated_batch":self.previous_generated_batch}).content
                        self.previous_generated_batch = res
                        cur_sample = try_parse_json(res)
                        cur_sample_dict = sample_str_to_dataframes_dict(cur_sample)
                        if results_dict=={}:
                            results_dict = cur_sample_dict
                        else:
                            for key, items in cur_sample_dict.items():
                                results_dict[key] = pd.concat([results_dict[key], items], ignore_index=True)
                        if leading_key == '':
                            leading_key = list(results_dict.keys())[0]
                        self.leading_key = leading_key
                        n = n + cur_sample_dict[leading_key].shape[0]
                        break  # Break out of the retry loop if successful
                    except Exception as inner_ex:
                        print(f"Error during generation (number of records {n}, retry {retries + 1}): {inner_ex}")
                        retries += 1
                        total_retries += 1

                        if total_retries >= total_max_retries:
                            print(f"Reached total maximum retries ({total_max_retries}). Exiting.")
                            return results_dict

            if (n > num_records):
                for key, items in results_dict.items():
                    results_dict[key] = items[:num_records]

            if output_format in [JSON_,STRING_]:
                results = dataframes_dict_to_string(results_dict)
            if output_format == JSON_:
                results = json.loads(results)
            if output_format == DATAFRAME_DICT_:
                results = results_dict

            return results
        except Exception as ex:
            print(f"Error during generation: {ex}")
            return results_dict

    async def async_generate(self, query, unique_id, randomness, max_retries=3):
        """
        Asynchronously generate synthetic data using the configured data generation chain.

        Parameters:
        - query (str, optional): The description of the required content.
        - unique_id (int): An identifier for the generation task.
        - randomness (float): A random value for the generation task.
        - max_retries (int, optional): Maximum number of retries for a single generation attempt. Defaults to 3.

        Returns:
        str: The generated synthetic data response.
        """
        retries = 0
        while retries < max_retries:
            try:
                if self.examples_data is not None:
                    cur_structure = self.examples_data.sample(n=self.batch_size)
                    leading_key = self.leading_key
                    cur_structure = dataframe_to_json(cur_structure, leading_key)
                else:
                    cur_structure = self.structure

                #response = await self.data_augmentor_chain.apredict(structure=cur_structure, randomness=randomness,
                #                                               human_input=query, previous_generated_batch = 'unknown')
                response = await self.data_augmentor_chain.ainvoke({"structure": cur_structure, "randomness": randomness,
                                                        "human_input": query,
                                                        "previous_generated_batch": 'unknown'})#.content
                return response
            except Exception as ex:
                print(f"Error during async generation (task {unique_id}, retry {retries + 1}): {ex}")
                retries += 1

        print(f"Reached maximum retries for async generation (task {unique_id}). Returning None.")
        return None

    async def generate_data_in_parallel(self, num_records, region=None, language=None, task_specifications=None, query='', max_retries=3, leading_key='', output_format = 0):
        """
        Generate synthetic data concurrently using multiple tasks.

        Parameters:
        - iterations (int): The number of concurrent data generation tasks.
        - region (str, optional): The required region. Defaults to an empty string.
        - language (str, optional): The required language. Defaults to an empty string.
        - query (str, optional): The description of the required content. Defaults to an empty string.
        - max_retries (int, optional): Maximum number of retries for a single generation attempt. Defaults to 3.

        Returns:
        List[str]: List of responses from the concurrent data generation tasks.

        Note:
        - This method generates synthetic data concurrently using asyncio tasks.
        - Each task generates data based on the provided parameters (query, region, language).
        - The method waits for all tasks to complete and aggregates the results.
        - If a task fails to generate data, it may perform multiple retries based on the max_retries parameter.
        """
        STRING_ = 0
        JSON_ = 1
        DATAFRAME_DICT_ = 2

        query_msg = compose_query_message(query, region, language, task_specifications)

        # Calculate the number of parallel tasks to run
        iterations = math.ceil(num_records / self.batch_size)
        results_dict = {}

        while iterations > 0:

            # Create asynchronous tasks for data generation
            tasks = [self.async_generate(query_msg, i, random.random(), max_retries) for i in range(1, iterations + 1)]

            # Execute tasks concurrently and gather results
            results = await asyncio.gather(*tasks)


            for i in range(len(results)):
                batch_str = try_parse_json(results[i].content)
                try:
                    batch_dataframes_dict = sample_str_to_dataframes_dict(batch_str)
                    if (results_dict == {}):
                        results_dict = batch_dataframes_dict
                    else:
                        for key, items in batch_dataframes_dict.items():
                            results_dict[key] = pd.concat([results_dict[key], items], ignore_index=True)
                except Exception as e:
                    print(f"Error processing batch number {i}: {e}")
                finally:
                    if leading_key == '':
                        leading_key = list(results_dict.keys())[0]
                    self.leading_key = leading_key
                    generated_records = results_dict[leading_key].shape[0]
                    iterations = math.ceil((num_records-generated_records) / self.batch_size)


        # Parse and aggregate results from concurrent data generation
        ##results = parse_content_generated_concurrently(results)

        # Remove additional records in the last batch
        if (generated_records > num_records):
            for key, items in results_dict.items():
                results_dict[key] = items[:num_records]

        if output_format in [JSON_, STRING_]:
            results = dataframes_dict_to_string(results_dict)
        if output_format == JSON_:
            results = json.loads(results)
        if output_format == DATAFRAME_DICT_:
            results = results_dict

        return results