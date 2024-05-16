from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from enum import Enum
from src.TaskSpecificationAugmentor import TaskSpecificationAugmentor


class PipelineName(Enum):
    DescriptionToMLDataset = 'DescriptionToMLDataset'
    DescriptionToDB = 'DescriptionToDB'
    SQLToTabular = 'SQLToTabular'
    DescriptionToUnstructured = 'DescriptionToUnstructured'
    ExamplesDataframeToTabular = 'ExamplesDataframeToTabular'
    APISpecificationToData = 'APISpecificationToData'
    UNKNOWN = 'UNKNOWN'

def get_pipeline_name(label):
    pipeline_mapping = {
        'DescriptionToMLDataset': PipelineName.DescriptionToMLDataset,
        'DescriptionToDB': PipelineName.DescriptionToDB,
        'SQLToTabular': PipelineName.SQLToTabular,
        'DescriptionToUnstructured': PipelineName.DescriptionToUnstructured,
        'ExamplesDataframeToTabular': PipelineName.ExamplesDataframeToTabular,
        'APISpecificationToData': PipelineName.APISpecificationToData,
        'UNKNOWN': PipelineName.UNKNOWN
    }

    if label in pipeline_mapping:
        return pipeline_mapping[label]
    else:
        raise ValueError("Invalid pipeline name")


class DataDefiner:
    """
    A class for extracting sample data structures based on user queries using a language model chain.
    """

    def __init__(self, llm, pipeline_name='', batch_size=10, verbose=True):
        """
        Initializes the DataDefiner.

        Parameters:
        - llm: The language model used for extraction.
        - pipeline_name: x2y description, x describes the type of input and y described the desired output
        - batch_size (int, optional): Number of examples to include in the extraction prompt. Defaults to 3.
        - verbose (bool, optional): Whether to print verbose output during extraction. Defaults to True.
        """

        data_definer_template = (
            f"You are a system that specializes in generating synthetic data according to user requests."
            f"Generate an initial synthetic data sample for the described task (according to the user guidance if provided, or your knowledge otherwise)."
            f"Provide a sample with {batch_size} number of items; items can be records in a table,records that can potentially be joined in a DB schema, or the equivalent tuples in a JSON format in unstructured text."
            f"The task as described by the user: {{human_input}};"
            f"The guidance given by an expert: {{extracted_specification}};"
            f"Format instructions: {self.get_pipeline_prompt(pipeline_name)}"
            f"Please make sure you output a valid JSON format, and don't cut it in the middle."
            f"Please make sure the output only contains the dataset structure. Omit any text before or after the JSON structure."
            f"\nGenerated Output:"
        )

        data_definer_prompt = PromptTemplate(
            input_variables=["human_input","extracted_specification"], template=data_definer_template
        )
        self.data_definer_chain = LLMChain(
            llm=llm,
            prompt=data_definer_prompt,
            verbose=verbose,
            output_key="structure",
        )

        self.batch_size = batch_size,
        self.pipeline_name = pipeline_name
        self.description = ''
        self.data_structure_sample = {}
        self.task_specifications = ''
        self.llm = llm

    def define_schema_from_description(self, description, task_specifications=None, max_trials=3):
        """
        Extract sample data structures based on a user description.

        Parameters:
        - description (str): The user's description of the task.
        - max_trials (int, optional): Maximum number of trials to attempt extraction. Defaults to 3.

        Returns:
        str: The predicted output containing a sample of data in the extracted structure.

        Note:
        - Performs multiple trials to handle extraction failures.
        """
        self.description = description
        self.task_specifications = task_specifications
        trial = 0
        while trial < max_trials:
            try:
                self.data_structure_sample  = self.data_definer_chain.predict(human_input = description, extracted_specification = self.task_specifications)
                return self.data_structure_sample
            except Exception as ex:
                print(f"Error during extraction (trial {trial + 1}): {ex}")
                trial += 1

        print(f"Reached maximum trials for extraction. Returning None.")
        return None

    def get_pipeline_prompt(self, pipeline_name):
        """
        Get the prompt template corresponding to a given pipeline name.

        Parameters:
        - pipeline_name (PipelineName): The pipeline name.

        Returns:
        str: The prompt template for the specified pipeline.
        """
        switch_dict = {
            PipelineName.DescriptionToMLDataset: """Generate Sample synthetic records for tabular data (preferably a single flattened table) that can be used for training a well-performing ML model for the task described in the data description.\
                Please include any relevant attribute you can think of, as we do not want to miss any feature that the user may find useful.\
                Format the output as JSON with the dataset name as key.""",
            PipelineName.DescriptionToDB: """Generate Sample synthetic records for relational DB (one or more tables).\
                Please follow any relevant distributions for the stated fields, as we want this data to be as valid and useful as possible for development and testing.\
                Format the output as JSON with each table name as key and in the nested level the feature names as keys.""",
            PipelineName.SQLToTabular: """Generate Sample synthetic data that fit to the structure given by the SQL command that appears in the data description.\
                Please follow any relevant distributions for the stated fields, as we want this data to be as valid and useful as possible for development and testing.\
                Format the output as JSON with each table name as key.""",
            PipelineName.DescriptionToUnstructured: """Generate Sample data, potentially unstructured, that fit to the data description.\
                Format the output as JSONL with the dataset name as key and each line contains a single sampled text in the appropriate format.""",
            PipelineName.ExamplesDataframeToTabular: """Generate Sample data that fit to the example structure given in the data description.\
                Format the output as a JSONL file, where each line contains a single sampled text.""",
            PipelineName.APISpecificationToData: """Check the input and output of the API that is mentioned in the user desription, and generate synthetic API calls (with valid input and output) that fit to the required structure of this API.\
                Format your output as JSONL, with the actual API name as key and each line containing the input and output of a synthetic API call.""",
            PipelineName.UNKNOWN: '',
        }
        return switch_dict.get(pipeline_name, '')
