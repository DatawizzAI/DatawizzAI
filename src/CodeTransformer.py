from langchain.prompts import PromptTemplate
from src.DataTransformer import DataTransformer
import random
import json
import pandas as pd
import asyncio
from src.utils.utils import replace_param_value, dataframes_dict_to_string

class CodeTransformer:
    """
    A class for extracting Python code for generating data based on expert specifications or user description using a language model.
    """

    def __init__(self,llm):
        """
        Initializes a new instance of the CodeTransformer class, which is designed to extract code for generating
        data and apply this code for data generation. This class uses a language model to generate code based
        on a user's description and applies this code to generate data.
        """
        self.code = ''
        self.results_dict = None
        self.table_size_param_dict = None
        self.override_fields_dict = None
        self.reformat_fields_dict = None
        self.table_names_dict = None
        self.primary_keys_dict = None
        self.parent_tables_dict = None
        self.cross_table_dependencies_dict = None
        self.llm = llm
        self.description=None
        self.specifications=None

    def generate_code_from_description(self, description, specifications, max_trials=4, verbose=True):
        """
        Extract Python code based on a user description or detailed specifications.

        Parameters:
        - llm: The language model used for extraction.
        - description (str): The description of the task.
        - specifications (str): The specifications of the data needed for the task.
        - max_trials (int, optional): Maximum number of trials to attempt extraction. Defaults to 3.
        - verbose (bool, optional): Whether to print verbose output during extraction. Defaults to True.

        Returns:
        str: The code generated using a language model.

        Note:
        - Performs multiple trials to handle extraction failures.
        """

        code_extraction_template = (
            f"You are a programmer whose job is to write Python code to generate data according to a given set of requirements, "
            f"making sure you don't miss any relevant detail, and handle all specified data characteristics, constraints and table relationships."
            f"1. Write the relevant imports (preferably use pandas, numpy, timestamp, datetime, random, and faker packages that are already installed). "
            f"2. Write parameters with the number of records to be genertated for each of the tables (initialized to up to 10 records per table, tables can have different sizes)."
            f"3. Write the code to generate parent tables (follow the recommended order of table generation if exists in the given data specification). All datetime fields should be set to utc timezone and converted to strings. Use datetime+timedelta(seconds=t) when handling potential negative timestamps. **All freetext fields** (as appears in the override_fields_dict parameter) **should contain empty values.** "
            f"4. Write the code to generate child / dependent tables, making sure you maintain valid keys, values and formats to guarantee referential integrity in the generation code itself and not in postprocessing steps. All datetime fields should be set to utc timezone, and converted to strings. Use datetime+timedelta(seconds=-t) when handling potential negative timestamps. All foreign keys should contain values that exist as primary keys in the parent tables. **All freetext fields** (as appears in the override_fields_dict parameter) **should contain empty values.** "
            f"5. Make sure all the specified formats of the fields are maintained (e.g. int vs. float). Table names should be identical to the names of the object that store them."
            f"6. If there exists any dependencies between datetime fields among the various tables, either make sure the previous code handles them, or write the code to enforce valid values in the datetime fields given these dependencies."
            f" E.g. add postprocessing commands to enforce date these fields dependencies: "
            f"sales_transactions['transaction_date'] = [fake.date_time_between_dates(datetime_start=pd.to_datetime(books.loc[books['book_id'] == book_id, 'publication_date'].values[0])) for book_id in sales_transactions['book_id']]"
            f"7. Write the code that defines the results_dict object with pairs of key (table name) and value (dataframe with the generated table)."
            "Order this dictionary according to the recommended order of table generation."
            f"8. Write the code that defines the table_size_param_dict object with pairs of key (table name) and value (it's number of records parameter name)."
            f"9. Write the code that defines the override_fields_dict dictionary with pairs of key (table name) and value (a list that includes all free text fields or categorical fields without a specified closed set of categories). Do not include tables with an empty list of fields."
            f"10. Write the code that defines the reformat_fields_dict dictionary with pairs of key (table name) and value (a list that includes all numerical and datetime fields except for primary key fields). Do not include tables with an empty list of fields."
            f"11. Write the code that defines the table_names_dict dictionary with pairs of key (table name (str)) and value (the name of the object storing this table (str))."
            f"12. Write the code that defines the primary_keys_dict object with pairs of key (table name) and value (primary key field). Order this dictionary according to the reccomended order of table generation."
            f"13. Write the code that defines the parent_tables_dict dictionary with pairs of key (table name) and value (a pair with (a) list of parent table names and (b) the Python command needed for joining the key table with all its parent tables). Do not include in the dictionary tables with no parent tables."
            f"14. Write the code that defines the cross_table_dependencies_dict dictionary with pairs of key (table name) and value (a TEXTUAL description of its Date/Time Fields dependencies with other fields). Do not include  in the dictionary tables that have no dependencies."    
            f"\nThe data specification as described by the expert: {{human_input}};"
            f" You MUST make sure you output a **valid and complete** Python code, just code, no intro and summary or prefix are needed, and don't cut it in the middle. "
            f"\nGenerated Extracted Code:"
        )
        self.description = description
        self.specifications = specifications

        code_extraction_prompt = PromptTemplate(input_variables=["human_input"], template=code_extraction_template)

        code_extraction_chain = code_extraction_prompt | self.llm #| StrOutputParser()

        trial = 0
        while trial < max_trials:
            try:
                current_code=''
                output_code = code_extraction_chain.invoke({"human_input":specifications})
                current_code = output_code.content.replace('```python', '').replace('```', '')
                loc = {}
                exec(current_code, globals(), loc)
                self.code = current_code
                self.results_dict = loc['results_dict']
                self.table_size_param_dict = loc['table_size_param_dict']
                self.override_fields_dict = loc['override_fields_dict']
                self.reformat_fields_dict = loc['reformat_fields_dict']
                self.table_names_dict = loc['table_names_dict']
                self.primary_keys_dict = loc['primary_keys_dict']
                self.parent_tables_dict = loc['parent_tables_dict']
                self.cross_table_dependencies_dict = loc['cross_table_dependencies_dict']

                return current_code
            except Exception as ex:
                print(f"Error during code extraction (trial {trial + 1}): {ex}")
                trial += 1
                current_code = self.autocorrect_code(self.llm, code=current_code, error_message=ex)
                if current_code is not None:
                    return current_code

        print(f"Reached maximum trials for code extraction. Returning None.")
        return None


    def autocorrect_code(self, llm, code, error_message, max_trials=5, verbose=True):
        """
        Extract Python code based on a user description or detailed specifications. By repeatively excecuting the code, catching errors, and correcting them.

        Parameters:
        - llm: The language model used for extraction.
        - code (str): The code that produces errors.
        - error_message (str): The error message.
        - max_trials (int, optional): Maximum number of trials to attempt extraction. Defaults to 3.
        - verbose (bool, optional): Whether to print verbose output during extraction. Defaults to True.

        Returns:
        str: The corrected code generated using a language model.

        Note:
        - Performs multiple trials to handle code execution failures.
        """

        code_correction_template = (
            f"You are a programmer whose job is to correct a Python code according to previous errors and your knowledge"
            f"Rewrite the input code so that it will be ** valid and complete**"
            f"\nThe given code: {{human_input}};"
            f"\nThe given error: {{error_message}};"
            f" You MUST make sure you output a **valid and complete** Python code, just code, no intro and summary or prefix are needed, and don't cut it in the middle."
            f"\nGenerated Extracted Code:"
        )

        code_correction_prompt = PromptTemplate(input_variables=["human_input", "error_message"], template=code_correction_template)

        code_correction_chain = code_correction_prompt | llm  # | StrOutputParser()
        trial = 0
        current_code = code
        current_error = error_message
        success = False
        while (success is False) and (trial < max_trials):
            try:
                success = True
                # task_specification = code_extraction_chain.predict(human_input=description)
                corrected_code = code_correction_chain.invoke({"human_input": current_code,"error_message": current_error})
                current_code = corrected_code.content
                loc={}
                exec(current_code, globals(), loc)
                self.code = current_code
                self.results_dict = loc['results_dict']
                self.table_size_param_dict = loc['table_size_param_dict']
                self.override_fields_dict = loc['override_fields_dict']
                self.reformat_fields_dict = loc['reformat_fields_dict']
                self.table_names_dict = loc['table_names_dict']
                self.primary_keys_dict = loc['primary_keys_dict']
                self.parent_tables_dict = loc['parent_tables_dict']
                self.cross_table_dependencies_dict = loc['cross_table_dependencies_dict']

                return current_code
            except Exception as ex:
                print(f"Error during auto correction ( trial {trial + 1}): {ex}")
                trial += 1
                current_error = ex
                success = False
                if ex.args[0]==22:
                    trial = max_trials

        print(f"Reached maximum trials for autocorrection. Returning None.")
        return None

    def enhance_tables_with_transformer(self, llm, description, max_trials=1, run_in_parallel=True, full_query=None):
        """
        Enhance tables created with Python code by utilizing pretrained LLMs to enhance tables (generating texts, improving formats ect.).

        Parameters:
        - llm: The language model used for extraction.
        - description (str): The description of the task.
        - max_trials (int, optional): Maximum number of trials to attempt extraction. Defaults to 3.

        Returns:
        A dictionary with the enhanced data: a dictionary with key (table name) and value (dataframe).

        Note:
        - Performs multiple trials to handle extraction failures.
            """
        # Loop over each table according to the extracted order (from independent to dependent tables)
        enriched_data = {}
        for tab_no in range(len(self.results_dict.keys())):
            override_fields_list = []
            restricted_fields_list = []
            tab_name = list(self.results_dict.keys())[tab_no]
            if tab_name not in enriched_data.keys():
                enriched_data[tab_name] = self.results_dict[tab_name]
            #print(tab_name)
            if (tab_name in self.override_fields_dict.keys()) and (len(self.override_fields_dict[tab_name])>0):
                override_fields_list = self.override_fields_dict[tab_name]
                if (len(override_fields_list)>0):
                    #enriched_data[tab_name][override_fields_list] = ''
                    enriched_data[tab_name].loc[:, override_fields_list] = ''
            if tab_name in self.primary_keys_dict.keys() and (len(self.primary_keys_dict[tab_name])>0):
                restricted_fields_list = self.primary_keys_dict[tab_name]
            # Joining dependent tables with their parent tables
            if (tab_name in self.parent_tables_dict.keys()) and (len(self.parent_tables_dict[tab_name][0])>0):
                merge_command = self.parent_tables_dict[tab_name][1]
                joined_table_name = self.parent_tables_dict[tab_name][0][0]
                merge_command = merge_command.replace(self.table_names_dict[tab_name],tab_name)
                merge_command = merge_command.replace(self.table_names_dict[joined_table_name],joined_table_name)
                trial = 0
                #results_df=None
                while trial < max_trials:
                    current_code = "import pandas as pd \n"
                    current_code =  current_code + 'results_df = ' + merge_command
                    current_code = current_code.replace(tab_name, 'enriched_data["' + tab_name + '"]')
                    current_code = current_code.replace(joined_table_name,'enriched_data["' + joined_table_name + '"]')
                    try:
                        loc = {}
                        #print(current_code)
                        exec(current_code, {"enriched_data": enriched_data}, loc)
                        tmp_merged_data = loc['results_df']
                        enriched_data[tab_name] = tmp_merged_data
                        restricted_fields_list = list(set([restricted_fields_list] + self.results_dict[joined_table_name].columns.values.tolist()))
                        trial += 1
                    except Exception as ex:
                        print(f"Error during code extraction (trial {trial + 1}): {ex}")
                        trial += 1
                        #CodeTransformerObj = CodeTransformer()
                        #current_code = CodeTransformerObj.autocorrect_code(code=current_code, error_message=ex, llm=llm)

            # else:
            # Define the transformation based on a user's description and apply it to the data.
            query = "Generate relevant and high quality unique, variable texts for free text fields: " + str(
                override_fields_list) + " Never override a value in a field from the following list: " + str(
                restricted_fields_list) + ". Reformat or correct any of the other numeric and datetime fields when needed (only change their values if mandatory for a valid record)."

            # If cross table constraint exist
            if tab_name in self.cross_table_dependencies_dict.keys():
                constraints_msg = str(self.cross_table_dependencies_dict[tab_name])
                query = query + " Correct the needed values to make sure these constraints are fulfilled: " + constraints_msg

            DataTransformerObj = DataTransformer(llm=llm)
            DataTransformerObj.extracted_logic = query
            DataTransformerObj.structure = '' ##
            full_description = description
            if full_query is not None:
                full_description = full_description +  full_query
            DataTransformerObj.description = full_description + "\n" + "Help me improve the quality and validity of the " + tab_name + " table. Ensure that the generated texts are distinct and varied, without repeating any names from previous requests, even across different sessions.\n" + "Do not return empty values unless you think the original value should be erased."

            ###tmp_result = DataTransformerObj.transform(source_data=enriched_data[tab_name],output_format=2)
            if run_in_parallel:
                tmp_result = asyncio.run(DataTransformerObj.transform_in_parallel(source_data=enriched_data[tab_name], output_format=2))
            else:
                tmp_result = DataTransformerObj.transform(source_data=enriched_data[tab_name], output_format=2)
            original_cols = self.results_dict[tab_name].columns.values.tolist()
            print('results:')
            print(tmp_result)
            ##enriched_data[tab_name] = pd.DataFrame(tmp_result[tab_name])[original_cols]
            enriched_data[tab_name] = tmp_result[original_cols]
        return enriched_data

    def generate_data(self, table_size_dict=None, max_trials=3, output_format=2, run_in_parallel=True, full_query = None):
        """
        Extract Python code based on a user description or detailed specifications.

        Parameters:
        - table_size_dict (dictionary): A dictionary with pairs of table name (key) and number of records to generate (value).
        - max_trials (int, optional): Maximum number of trials to attempt extraction. Defaults to 3.

        Returns:
        str: The updated code and the tables generated.

        Note:
        - Performs multiple trials to handle execution failures.
        """
        STRING_ = 0
        JSON_ = 1
        DATAFRAME_DICT_ = 2

        current_code = self.code
        if table_size_dict is not None:
            for param_name, new_param_value in table_size_dict.items():
                current_code = replace_param_value(current_code, str(self.table_size_param_dict[param_name]), new_param_value)
        trial = 0
        while trial < max_trials:
            try:
                loc = {}
                exec(current_code, globals(), loc)
                self.code = current_code
                self.results_dict = loc['results_dict']
                self.table_size_param_dict = loc['table_size_param_dict']
                self.override_fields_dict = loc['override_fields_dict']
                self.reformat_fields_dict = loc['reformat_fields_dict']
                self.table_names_dict = loc['table_names_dict']
                self.primary_keys_dict = loc['primary_keys_dict']
                self.parent_tables_dict = loc['parent_tables_dict']
                self.cross_table_dependencies_dict = loc['cross_table_dependencies_dict']

                results = self.enhance_tables_with_transformer(llm=self.llm,description=self.description, run_in_parallel=run_in_parallel, full_query=full_query)

                if output_format in [JSON_, STRING_]:
                    results = dataframes_dict_to_string(results)
                if output_format == JSON_:
                    results = json.loads(results)

                return results #self.code
            except Exception as ex:
                print(f"Error during data generation from code (trial {trial + 1}): {ex}")
                trial += 1
                #current_code = self.autocorrect_code(self.llm, code=current_code, error_message=ex)
                return None #current_code

        print(f"Reached maximum trials for extraction. Returning None.")
        return None



