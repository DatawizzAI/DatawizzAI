from langchain.prompts import PromptTemplate
from src.Pipeline import *

class DataDefiner:
    """
    A class for extracting sample data structures based on user queries using a language model chain.
    """

    def __init__(self, llm, pipeline_name=None, batch_size=10, verbose=True):
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
            f"Format instructions: {Pipeline.get_pipeline_prompt(pipeline_name)}"
            f"Please make sure you output a valid JSON format, and don't cut it in the middle."
            f"Please make sure the output only contains the dataset structure. Omit any text before or after the JSON structure."
            f"\nGenerated Output:"
        )

        data_definer_prompt = PromptTemplate(
            input_variables=["human_input","extracted_specification"], template=data_definer_template
        )

        self.data_definer_chain = data_definer_prompt | llm

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
                #self.data_structure_sample  = self.data_definer_chain.predict(human_input = description, extracted_specification = self.task_specifications)
                self.data_structure_sample = self.data_definer_chain.invoke({"human_input":description,"extracted_specification":self.task_specifications}).content

                return self.data_structure_sample#.content
            except Exception as ex:
                print(f"Error during extraction (trial {trial + 1}): {ex}")
                trial += 1

        print(f"Reached maximum trials for extraction. Returning None.")
        return None
