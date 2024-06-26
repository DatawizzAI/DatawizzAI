from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class TaskSpecificationAugmentor:
    """
    A class for extracting expert guidelines and specifications based on user description using a language model.
    """

    def generate_specifications_from_description(llm, description, max_trials=3, verbose=True):
        """
        Extract task specifications based on a user description.

        Parameters:
        - llm: The language model used for extraction.
        - description (str): The user's description of the task.
        - max_trials (int, optional): Maximum number of trials to attempt extraction. Defaults to 3.
        - verbose (bool, optional): Whether to print verbose output during extraction. Defaults to True.

        Returns:
        str: The predicted output containing a sample of data in the extracted structure.

        Note:
        - Performs multiple trials to handle extraction failures.
        """
        specification_extraction_template = (
            f"You are an analyst whose job is to conduct a deep research for a given task, and define the needed data to collect, "
            f"making sure you don't miss any relevant detail, and gain a deep understanding of the data characteristics."
            f"1.  Generate the names of the columns relevant for the description of the user. Use indicative names. "
            f"Assign each column its type - numerical, categorical, datetime, free text, unique identifier. "
            f"2. Revisit each column and complete these details:"
            f"For numeric columns - describe its distribution, mean and std, min and max values. "
            f"for numbers and datetimes define the needed format, for categorical columns detail a complete set of categories and its probabilities. "
            f"For free text columns - specify the mean and std of the text length, "
            f"For unique identifier columns - specify the format and regEX to follow. "
            f"For Datetime columns - specify min and max values, as well as the time intervals mean and std values. "
            f"3. Important correlations to maintain or other constraints and dependencies. "
            f"4. Indicate what fields would be better to be generated by a LLM instead of by a simple code. "
            f"5. Indicate if this table is has sequential (transactional) nature. "
            f"\nThe task as described by the user: {{human_input}};"
            f"Please make sure you output a valid textual description, just guidance, no intro and summary are needed, and don't cut it in the middle."
            f"\nGenerated Extracted Logic:"
        )

        specification_extraction_prompt = PromptTemplate(
            input_variables=["human_input"], template=specification_extraction_template
        )

        specification_extraction_chain = LLMChain(llm=llm, prompt=specification_extraction_prompt, verbose=verbose)
        # self.description = description

        trial = 0
        while trial < max_trials:
            try:
                task_specification = specification_extraction_chain.predict(human_input=description)
                return task_specification
            except Exception as ex:
                print(f"Error during extraction (trial {trial + 1}): {ex}")
                trial += 1

        print(f"Reached maximum trials for extraction. Returning None.")
        return None

    def refine_specifications_by_description(llm, description, previous_task_specification, max_trials=3, verbose=True):
        """
        Extract task specifications based on a user description.

        Parameters:
        - llm: The language model used for extraction.
        - description (str): The user's description of the task.
        - max_trials (int, optional): Maximum number of trials to attempt extraction. Defaults to 3.
        - verbose (bool, optional): Whether to print verbose output during extraction. Defaults to True.

        Returns:
        str: The predicted output containing a sample of data in the extracted structure.

        Note:
        - Performs multiple trials to handle extraction failures.
        """
        specification_extraction_template = (
            f"You are an analyst whose job is to conduct a deep research for a given task, and define the needed data to collect, "
            f"making sure you don't miss any relevant detail, and gain a deep understanding of the data characteristics."
            f"You already gave instructions for the needed data (see Previous Task Specifications), but now the user asks a content refinement (see User Query). "
            f"Please revisit the columns distributions, and descriptive statistics and update those that have changed due to the user query. "
            f"\nThe User Query: {{human_input}};"
            f"\nYour Previous Task Specifications: {{previous_specifications}};"
            f"Please make sure you output a valid textual description, just guidance, no intro and summary are needed, and don't cut it in the middle."
            f"\nGenerated Extracted Logic:"
        )

        specification_extraction_prompt = PromptTemplate(
            input_variables=["human_input", "previous_specifications"], template=specification_extraction_template
        )

        specification_extraction_chain = LLMChain(llm=llm, prompt=specification_extraction_prompt, verbose=verbose)
        # self.description = description

        trial = 0
        while trial < max_trials:
            try:
                task_specification = specification_extraction_chain.predict(human_input=description,
                                                                            previous_specifications=previous_task_specification)
                return task_specification
            except Exception as ex:
                print(f"Error during extraction (trial {trial + 1}): {ex}")
                trial += 1

        print(f"Reached maximum trials for extraction. Returning None.")
        return None

