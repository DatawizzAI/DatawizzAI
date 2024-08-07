from langchain.prompts import PromptTemplate
import json


class TaskSpecificationAugmentor:
    """
    A class for extracting expert guidelines and specifications based on user description using a language model.
    """
    def __init__(self, llm, batch_size=10, verbose=True):
        """
        Initializes a new instance of the TaskSpecificationAugmentor class, which is designed to extract task specification
        for a given task. This class uses a language model to generate specifications based
        on a user's description of the task.

        Parameters:
        - llm (LLM): The language model used for generating transformation logic.

        """
        self.batch_size = batch_size
        self.verbose = verbose
        self.llm = llm
        self.description = ''

        self.specification_extraction_template = (
            f"You are an analyst whose job is to conduct a deep research for a given task, and specify the needed data to collect or produce, "
            f"making sure you don't miss any relevant detail, and gain a deep understanding of the data characteristics."
            f"1. Generate the names of the needed tables (one table or more)."
            f"2. For each table, generate the names of a rich set of relevant columns. Use indicative names. "
            f"Assign each column its type - integer, float, categorical, datetime, free text, and it's special role: primary key or foreign key. "
            f"3. Revisit each column on each of the tables and complete the following details:"
            f"  a. For numeric columns - ALWAYS specify its distribution, and its mean, std, min, max values. Also,specify the percentage of empty values"
            f"  b. For categorical columns ALWAYS specify a complete set of categories **AND** its probabilities (numbers between 0-1), and the percentage of empty values. "
            f"  c. For unique identifier columns - ALWAYS specify the format, min and max values, and regEX to follow. "
            f"  d. For Datetime columns - ALWAYS specify min and max values, **and** the time intervals mean and std values. "
            f"  e. For numbers and date/time columns ALSO define the needed format. "
            f"4. FOR EACH TABLE specify the following:"
            f"  a. A comma seperated list of pairs of highly correlated fields (potentially explaining each other) and specify the correlation. "
            f"  b. A comma seperated list of free text fields."
            f"  c. A comma seperated list of date / time fields. "
            f"  d. Indicate if the table has sequential (transactional) nature or not. "
            f"5. Eventually extract some cross table insights:"
            f"  a. Dictate the recommended order of table generation, from less dependent table, to most dependent table. "
            f"  b. State highly correlated fields across tables (state both table and field names). Don't mention correlation with primary key fields in this list. "
            f"  c. Look for dependencies between pairs of datetime fields among the various tables (e.g. transaction_date should be greater than product_creation_date) and specify a list of such dependencies."
            f"\nThe task as described by the user: {{human_input}};"
            f"Please make sure you output a valid textual description in English (categories can be in other languages), just guidance, no intro and summary are needed, and don't cut it in the middle."
            f"\nGenerated Extracted Specifications:"
        )

    def generate_specifications_from_description(self, description, max_trials=3):
        """
        Extract task specifications based on a user description.

        Parameters:
        - llm: The language model used for extraction.
        - description (str): The user's description of the task.
        - max_trials (int, optional): Maximum number of trials to attempt extraction. Defaults to 3.

        Returns:
        str: The predicted output containing a sample of data in the extracted structure.

        Note:
        - Performs multiple trials to handle extraction failures.
        """

        self.description= description

        specification_extraction_prompt = PromptTemplate(
            input_variables=["human_input"], template=self.specification_extraction_template
        )

        #specification_extraction_chain = LLMChain(llm=llm, prompt=specification_extraction_prompt, verbose=verbose)
        specification_extraction_chain = specification_extraction_prompt | self.llm #| StrOutputParser()

        trial = 0
        while trial < max_trials:
            try:
                # Generate task specifications
                task_specification = specification_extraction_chain.invoke({"human_input":self.description})

                # Score the generated specifications
                specifications_evaluation = self.validate_task_specifications(
                    specifications=task_specification.content)
                specifications_evaluation = json.loads(specifications_evaluation.content)
                # Loop to auto correct the specifications (correct & evaluate in each iteration)
                score = specifications_evaluation['score']
                errors = specifications_evaluation['errors']
                print("Specification's score: %d" % (score))

                internal_trial = 0
                while score < 100 and internal_trial < max_trials:
                    print("Autocorrecting task specifications:")
                    task_specification = self.correct_task_specifications(
                        specifications=task_specification.content, errors=errors)
                    corrected_specifications_evaluation = self.validate_task_specifications(
                        specifications=task_specification.content)
                    #corrected_specifications_evaluation = try_parse_json(corrected_specifications_evaluation.content)
                    corrected_specifications_evaluation = json.loads(corrected_specifications_evaluation.content)
                    errors = corrected_specifications_evaluation['errors']
                    score = corrected_specifications_evaluation['score']
                    print(f"Specification's score: {score} ; The following errors were detected:\n {errors} ")
                    #print(str(internal_trial) + ':' + str(score) + ':' + str(errors))
                    internal_trial = internal_trial + 1

                return {'task_specifications': task_specification.content, 'score':score, 'errors':errors}
            except Exception as ex:
                print(f"Error during task specifications extraction (trial {trial + 1}): {ex}")
                trial += 1

        print(f"Reached maximum trials for task specifications extraction. Returning None.")
        return None

    def refine_specifications_by_description(self, description, previous_task_specification, max_trials=3, verbose=True):
        """
        Extract task specifications based on a user description.

        Parameters:
        - description (str): The user's description of the task.
        - max_trials (int, optional): Maximum number of trials to attempt extraction. Defaults to 3.
        - previous_task_specification (str): the input specifications to be refined

        Returns:
        str: The predicted output containing a sample of data in the extracted structure.

        Note:
        - Performs multiple trials to handle extraction failures.
        """
        self.description= description
        specification_extraction_template = (
            f"You are an analyst whose job is to conduct a deep research for a given task, and define the needed data to collect, "
            f"making sure you don't miss any relevant detail, and gain a deep understanding of the data characteristics."
            f"You already gave instructions for the needed data (see Previous Task Specifications), but now the user asks a content refinement (see User Query). "
            f"Please revisit the columns distributions, and descriptive statistics and update those that have changed due to the user query. "
            f"\nThe User Query: {{human_input}};"
            f"\nYour Previous Task Specifications: {{previous_specifications}};"
            f"Please make sure you output a valid textual description in English (categories can be in other languages), just guidance, no intro and summary are needed, and don't cut it in the middle."
            f"\nGenerated Extracted Logic:"
        )

        specification_extraction_prompt = PromptTemplate(
            input_variables=["human_input", "previous_specifications"], template=specification_extraction_template
        )

        #specification_extraction_chain = LLMChain(llm=self.llm, prompt=specification_extraction_prompt, verbose=self.verbose)
        specification_extraction_chain = specification_extraction_prompt | self.llm
        # self.description = description

        trial = 0
        while trial < max_trials:
            try:
                #task_specification = specification_extraction_chain.predict(human_input=self.description,
                #                                                            previous_specifications=previous_task_specification)
                task_specification = specification_extraction_chain.invoke(
                    {"human_input": self.description, "previous_specifications": previous_task_specification}).content
                return task_specification
            except Exception as ex:
                print(f"Error during extraction (trial {trial + 1}): {ex}")
                trial += 1

        print(f"Reached maximum trials for extraction. Returning None.")
        return None

    def validate_task_specifications(self, specifications, max_trials=3):
        """
        Extract a list of missing / erronious details in a task specifications based on a user description and task_specification.

        Parameters:
        - description (str): The user's description of the task.
        - specifications (str): The llm's generated task specifications that we wish to validate.
        - max_trials (int, optional): Maximum number of trials to attempt extraction. Defaults to 3.
        Returns:
        str: A list of gaps / errors found within the given task specifications.

        Note:
        - Performs multiple trials to handle extraction failures.
        - Returns None if no gaps were found.
        """

        specification_validation_template = (
            f"You are an analyst aiming to complete your task in the most professional and COMPLETE way. You already gave instructions for the needed data (see 'Your Previous Output') and now you want to check it in order to make it perfect. "
            f"1. Go over the given task (see Your Task below) and the user description of it (see User Description below) and CAREFULLY check your previous output and list any detail you were required to describe and missed or got wrong or partial"
            f"(e.g. missing fields, missing categories, missing distributions, missing values for means, X or dots instead of numbers, etc.). "
            f"2. Score your previous output with an integer between 0 and 100 according to the percentage of detected gaps and errors out of the overall needed details in this task."
            f"\nUser Description: {{human_input}};"
            f"\nYour Task: "+self.specification_extraction_template+";"
            f"\nYour Previous Output: {{latest_instructions}};"
            f"Please make sure you output a valid dictionary with 2 items: 'score' (the numeric score mentioned above), and 'errors' (a list of strings, each describe an error or gap detected within your previous output). "
            f"Make sure you don't cut it in the middle and ALWAYS output a valid dictionary with these exact keys. "
            f"\nGenerated Validation Output:"
        )


        specification_validation_prompt = PromptTemplate(
            input_variables=["human_input","latest_instructions"], template=specification_validation_template
        )

        #specification_extraction_chain = LLMChain(llm=llm, prompt=specification_extraction_prompt, verbose=verbose)
        specification_validation_chain = specification_validation_prompt | self.llm #| StrOutputParser()

        trial = 0
        while trial < max_trials:
            try:
                task_specification = specification_validation_chain.invoke({"human_input":self.description,"latest_instructions":specifications})
                return task_specification
            except Exception as ex:
                print(f"Error during task specifications evaluation (trial {trial + 1}): {ex}")
                trial += 1

        print(f"Reached maximum trials for task specifications evaluation. Returning None.")
        return None

    def correct_task_specifications(self, specifications, errors, max_trials=3):
        """
        Extract a list of missing / erronious details in a task specifications based on a user description and task_specification.

        Parameters:
        - specifications (str): The llm's generated task specifications that we wish to validate.
        - max_trials (int, optional): Maximum number of trials to attempt extraction. Defaults to 3.
        - verbose (bool, optional): Whether to print verbose output during extraction. Defaults to True.

        Returns:
        str: A list of gaps / errors found within the given task specifications.

        Note:
        - Performs multiple trials to handle extraction failures.
        - Returns None if no gaps were found.
        """


        specification_correction_template = (
            f"You are an analyst aiming to complete your task in the most professional and COMPLETE way. You already gave instructions for the needed data (see 'Your Previous Specifications'). "
            f"1. Go over the given task (see Your Task below) and the user description of it (see User Description below) and CAREFULLY check the your previous specifications. "
            f"2. Rewrite your previous specifications as follows: Scan the list of detected gaps and errors (see 'Errors' below) and correct/complete all the listed errors and gaps."
            f"(DON'T omit or change any of the valid parts). Group together related details if needed, to better organize the output."            
            f"\nUser Description: {{human_input}};"
            f"\nYour Task: "+self.specification_extraction_template+";"
            f"\nYour Previous Specifications: {{latest_instructions}};"
            f"\nDetected Errors: {{errors}};"
            f"Please make sure you output a valid textual description, just the revised specifications, no intro and summary are needed, and don't cut it in the middle."
            f"\nGenerated Your Revised Specifications:"
        )

        specification_validation_prompt = PromptTemplate(
            input_variables=["human_input","latest_instructions","errors"], template=specification_correction_template
        )

        #specification_extraction_chain = LLMChain(llm=llm, prompt=specification_extraction_prompt, verbose=verbose)
        specification_validation_chain = specification_validation_prompt | self.llm #| StrOutputParser()

        trial = 0
        while trial < max_trials:
            try:
                task_specification = specification_validation_chain.invoke({"human_input":self.description,"latest_instructions":specifications,"errors":errors})
                return task_specification
            except Exception as ex:
                print(f"Error during task specifications correction (trial {trial + 1}): {ex}")
                trial += 1

        print(f"Reached maximum trials for task specifications correction. Returning None.")
        return None