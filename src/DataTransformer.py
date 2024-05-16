from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from src.utils.utils import *
import asyncio


class DataTransformer:
    def __init__(self, llm, src_data=None, batch_size=10, verbose=True):
        """
        Initializes a new instance of the DataTransformer class, which is designed to extract rules for transforming
        data and apply these transformations. This class uses a language model to generate transformation logic based
        on a user's description and applies this logic to transform source data.

        Parameters:
        - llm (LLM): The language model used for generating transformation logic.
        - src_data (dict of pandas.DataFrame, optional): Initial source data to be transformed. Default is None.
        """
        self.batch_size = batch_size
        self.verbose = verbose
        self.src_data = src_data
        self.extracted_logic = None
        self.structure = ''

        # Template for extracting logic using the language model.
        logic_extraction_template = (
            "You are a system that specializes in enriching given source data with additional attributes "
            "according to user requests. Extract a set of rules to support the task described by the user. "
            "Utilize the user guidance if provided, and improve it with your own knowledge to a concise yet "
            "informative set of rules. These rules will be sent to you later in the conversation so only include rules "
            "that you can follow while performing the transformation. Remember that you can only see a small sample of the "
            "source data (around 10 records) so don't include rules that are based on training ML models on the given data. "
            "If the transformation involves adding new fields, please specify their names. "
            "The task as described by the user: {human_input}; "
            "The given source data sample: {src_data}; "
            "Please make sure you output a valid textual description, just guidance, no intro and summary are needed, "
            "and don't cut it in the middle."
        )

        logic_extraction_prompt = PromptTemplate(
            input_variables=["human_input", "src_data"], template=logic_extraction_template
        )

        # Chain for extracting logic from user input.
        self.logic_extraction_chain = LLMChain(llm=llm, prompt=logic_extraction_prompt, output_key="extracted_logic")

        # Template for transforming data based on the extracted logic.
        data_transformer_template = (
            "You are a system that specializes in enriching or transforming given source data with additional attributes "
            "according to user requests. Follow the given transformation logic for adding additional features. "
            "The task as described by the user: {human_input}; "
            "The transformation logic: {extracted_logic}; "
            "The output should be formatted as the input sample. "
            "The given source data is: {src_data}; "
            "Please make sure you output a valid JSON format, and don't cut it in the middle. "
            "Please make sure the output only contains the dataset structure. Omit any text before or after the JSON structure."
        )

        data_transformer_prompt = PromptTemplate(
            input_variables=["human_input", "src_data", "extracted_logic"], template=data_transformer_template
        )

        # Chain for applying transformations based on the extracted logic.
        self.data_transformer_chain = LLMChain(
            llm=llm,
            prompt=data_transformer_prompt,
            verbose=True,
            output_key="transformed_data"
        )

        # Sequential chain combining logic extraction and data transformation.
        self.overall_chain = SequentialChain(
            chains=[self.logic_extraction_chain, self.data_transformer_chain],
            input_variables=["human_input", "src_data"],
            output_variables=["extracted_logic", "transformed_data"],
            verbose=True
        )


    def define_transformation(self, source_data, description, max_trials=3):
        """
        Attempts to define a data transformation by extracting rules and applying them to provided source data.
        It tries multiple times if necessary to ensure a successful extraction.

        Parameters:
        - source_data (dict of pandas.DataFrame): The source data to be transformed.
        - description (str): A user-provided description of the transformation task.
        - max_trials (int): The maximum number of trials to attempt for successful rule extraction.

        Returns:
        - str: A JSON string representing a sample of the transformed data structure, or None if unsuccessful.
        """
        trial = 0
        while trial < max_trials:
            try:
                sample_data = create_json_sample_from_dataframes_dictionary(dataframes_dictionary=source_data)
                sample_data_transformed = self.overall_chain.invoke({"human_input": description, "src_data": sample_data})
                self.src_data = source_data
                self.extracted_logic = sample_data_transformed['extracted_logic']
                self.structure = sample_data_transformed['transformed_data']
                self.description = description
                return sample_data_transformed['transformed_data']
            except Exception as ex:
                print(f"Error during extraction (trial {trial + 1}): {ex}")
                trial += 1
        print("Reached maximum trials for extraction. Returning None.")
        return None

    def transform(self, source_data, batch_size=10, leading_key=''):
        """
        Transforms the source data in batches based on previously extracted transformation logic.

        Parameters:
        - source_data (dict of pandas.DataFrame): The source data to be transformed.
        - batch_size (int): The number of records to process in each batch.
        - leading_key (str, optional): The primary key used to align the batches. If not provided, the first key found is used.

        Returns:
        - dict: A dictionary containing the transformed data.
        """
        if not source_data:
            return {}

        leading_key = leading_key or next(iter(source_data), None)
        if leading_key is None:
            return {}  # Handle the case of empty input data

        output_data = {key: [] for key in source_data}
        total_records = len(source_data[leading_key])

        for start_idx in range(0, total_records, batch_size):
            end_idx = min((start_idx + batch_size), total_records)
            sample_data = {key: data[start_idx:end_idx].to_dict('records') for key, data in source_data.items() if
                           isinstance(data, pd.DataFrame)}

            try:
                sample_json = json.dumps(sample_data)
                transformed_json_str = self.data_transformer_chain.predict(
                    human_input=self.description, src_data=sample_json, extracted_logic=self.extracted_logic)
                transformed_data = json.loads(transformed_json_str)

                for key, items in transformed_data.items():
                    output_data[key].extend(items)
            except Exception as e:
                print(f"Error processing data from index {start_idx} to {end_idx}: {e}")
                continue
        return output_data

    async def transform_in_parallel(self, source_data, batch_size=10, leading_key=''):
        """
        Asynchronously transforms the source data in batches based on previously extracted transformation logic.
        This function uses concurrency to process different chunks of the data in parallel.

        Parameters:
        - source_data (dict of pandas.DataFrame): The source data to be transformed.
        - batch_size (int): The number of records to process in each batch.
        - leading_key (str, optional): The primary key used to align the batches. If not provided, the first key found is used.

        Returns:
        - dict: A dictionary containing the transformed data.
        """
        if not source_data:
            return {}

        leading_key = leading_key or next(iter(source_data), None)
        if leading_key is None:
            return {}  # Handle the case of empty input data

        output_data = {key: [] for key in source_data}
        total_records = len(source_data[leading_key])

        # Preparing batch processing tasks
        tasks = [self.process_batch(source_data, leading_key, start_idx, min(start_idx + batch_size, total_records))
                 for start_idx in range(0, total_records, batch_size)]

        # Execute tasks concurrently and gather results
        results = await asyncio.gather(*tasks)

        # Combine results from all the asynchronous tasks
        for result in results:
            if result:
                for key, items in result.items():
                    output_data[key].extend(items)

        return output_data

    async def process_batch(self, source_data, leading_key, start_idx, end_idx):
        """
        Process a single batch of data asynchronously. Converts DataFrame to JSON, sends it for processing,
        and then integrates the results.

        Parameters:
        - source_data (dict of pandas.DataFrame): The complete source data.
        - leading_key (str): The key used for batch alignment.
        - start_idx (int): Starting index of the batch.
        - end_idx (int): Ending index of the batch.

        Returns:
        - dict: A dictionary containing the transformed data for the batch.
        """
        # Extract batch data and convert to JSON
        batch_data = {key: df[start_idx:end_idx].to_dict('records') if key == leading_key else df for key, df in
                      source_data.items()}
        sample_json = json.dumps(batch_data)

        try:
            # Assuming predict is an asynchronous function
            transformed_json_str = await self.data_transformer_chain.apredict(
                human_input=self.description, src_data=sample_json, extracted_logic=self.extracted_logic)
            transformed_data = json.loads(transformed_json_str)
            return transformed_data
        except Exception as e:
            print(f"Error processing data from index {start_idx} to {end_idx}: {e}")
            return {}
