from langchain.prompts import PromptTemplate
#from langchain.chains import LLMChain#, SequentialChain
from src.Pipeline import *
from src.TaskSpecificationAugmentor import TaskSpecificationAugmentor
from src.DataDefiner import DataDefiner
from src.DataAugmentor import DataAugmentor
from src.CodeTransformer import CodeTransformer
from src.utils.utils import try_parse_json, sample_str_to_dataframes_dict, compose_query_message
#import pandas as pd
import json
import asyncio
#import nest_asyncio


class DataGenerationPipeline:
    """
    A class for wrapping end-to-end data generation tasks.
    """

    def __init__(self, llm, pipeline_name='', batch_size=10):
        """
        Initializes the DataPipeline.

        Parameters:
        - llm: The language model used for the various tasks.
        - pipeline_name: x2y description, x describes the type of input and y described the desired output
        - batch_size (int, optional): Number of examples to include in the extraction prompt. Defaults to 3.
        """


        pipeline_extractor_template = (
            f"You are a system that specializes in generating synthetic data according to user requests.\n"
            f"Given the optional pipelines (see Pipelines below), what is the needed pipeline for the data description given by the user (see User Task Description below).\n"
            f"The User Task Description: {{human_input}};\n"
            f"The optional Pipelines: \n"+ str(Pipeline.pipelines_descriptions) +";\n"
            f"The output should contain the pipeline name only, no additional description is needed.\n"
            f"\nExtracted pipeline:"
        )

        pipeline_extractor_prompt = PromptTemplate(input_variables=["human_input"], template=pipeline_extractor_template)

        self.pipeline_extractor_prompt = pipeline_extractor_prompt
        self.data_structure_sample = '',
        self.batch_size = batch_size,
        self.pipeline_name = pipeline_name
        self.description = ''
        self.task_specifications = ''
        self.llm = llm
        self.pipeline_extractor_chain = pipeline_extractor_prompt | self.llm
        self.code = ''




    def extract_pipeline_from_description(self, description, max_trials=3):
        """
        Extract pipeline name based on a user description of the task.

        Parameters:
        - description (str): The user's description of the task.
        - max_trials (int, optional): Maximum number of trials to attempt extraction. Defaults to 3.

        Returns:
        str: The predicted output containing the name of the extracted pipeline.

        Note:
        - Performs multiple trials to handle extraction failures.
        """
        self.description = description
        trial = 0
        while trial < max_trials:
            try:
                output = self.pipeline_extractor_chain.invoke({"human_input":description})
                self.pipeline_name = Pipeline.get_pipeline_name(output.content)

                return self.pipeline_name
            except Exception as ex:
                print(f"Error during pipeline extraction (trial {trial + 1}): {ex}")
                trial += 1

        print(f"Reached maximum trials for pipeline extraction. Returning None.")
        return None

    def extract_sample_data(self, description, pipelineName=None, outputFormat=0):
        STRING_ = 0
        JSON_ = 1
        DATAFRAME_DICT_ = 2


        if pipelineName is None:
            cur_pipeline = Pipeline.UNKNOWN
        else:
            cur_pipeline = pipelineName
        self.pipeline_name = cur_pipeline

        # Automatically extracting the pipeline if not given as a known input
        if (cur_pipeline == Pipeline.UNKNOWN):
            cur_pipeline = self.extract_pipeline_from_description(description=description)
            self.pipeline_name = cur_pipeline

        if cur_pipeline in [Pipeline.DescriptionToDB]:
            # Generate professional specifications
            TaskSpecificationAugmentorObj = TaskSpecificationAugmentor(llm=self.llm)
            task_specifications = TaskSpecificationAugmentorObj.generate_specifications_from_description(
                description=description)
            self.task_specifications = task_specifications['task_specifications']

            DataDefinerObj = DataDefiner(self.llm, pipeline_name=self.pipeline_name)
            self.data_structure_sample = DataDefinerObj.define_schema_from_description(description=description,
                                                                                       task_specifications=self.task_specifications)
        elif cur_pipeline == Pipeline.ExamplesDataframeToTabular:
            self.data_structure_sample = description
        else:
            DataDefinerObj = DataDefiner(self.llm, pipeline_name=self.pipeline_name)
            self.data_structure_sample = DataDefinerObj.define_schema_from_description(description=description)

        dataStructureSample = self.data_structure_sample
        if outputFormat in [JSON_]:
            dataStructureSample = try_parse_json(dataStructureSample)
            dataStructureSample = json.loads(dataStructureSample)
        if outputFormat in [DATAFRAME_DICT_]:
            dataStructureSample = try_parse_json(dataStructureSample)
            dataStructureSample = sample_str_to_dataframes_dict(dataStructureSample)
        return dataStructureSample

    def query_sample_data(self, query, region='', language='', outputFormat=0):
        STRING_ = 0
        JSON_ = 1
        DATAFRAME_DICT_ = 2

        cur_pipeline = self.pipeline_name
        # Automatically extracting the pipeline if not given as a known input
        if (cur_pipeline == Pipeline.UNKNOWN) or (self.data_structure_sample == ''):
            print("Please run method '''extract_sample_data''' first")
            return()

        full_query = compose_query_message(query=query, region=region, language=language)
        self.query = full_query

        if cur_pipeline in [Pipeline.DescriptionToDB]:
            if (len(self.task_specifications)==0):
                print("Please run method '''extract_sample_data''' first")
            # Generate professional specifications
            TaskSpecificationAugmentorObj = TaskSpecificationAugmentor(llm=self.llm)
            TaskSpecificationAugmentorObj.description = self.description
            self.task_specifications = TaskSpecificationAugmentorObj.refine_specifications_by_description(description = full_query, previous_task_specification=self.task_specifications, max_trials=3, verbose=True)
            #self.description = self.description + "; " + full_query

        DataAugmentorObj = DataAugmentor(llm=self.llm, structure=self.data_structure_sample)
        self.data_structure_sample = DataAugmentorObj.generate_data(query=query, region=region, language=language,
                                                                        task_specifications=self.task_specifications,output_format=0)

        dataStructureSample = self.data_structure_sample
        if outputFormat in [DATAFRAME_DICT_, JSON_]:
            dataStructureSample = try_parse_json(dataStructureSample)
            dataStructureSample = json.loads(dataStructureSample)
        if outputFormat in [DATAFRAME_DICT_]:
            #json_data = json.loads(dataStructureSample)
            dataStructureSample = sample_str_to_dataframes_dict(self.data_structure_sample)
        return dataStructureSample


    def generate_data(self, num_records=0, tables_size_dict=None, output_format=2, code = '', run_in_parallel=True, examples_dataframe_dict = None, query=None, region=None, language=None):
        STRING_ = 0
        JSON_ = 1
        DATAFRAME_DICT_ = 2

        self.code = code

        generated_data = None
        cur_pipeline = self.pipeline_name
        # Automatically extracting the pipeline if not given as a known input
        if (cur_pipeline == Pipeline.UNKNOWN) or (self.data_structure_sample == ''):
            print("Please run method '''extract_sample_data''' first")
            return()

        if (num_records == 0) and (tables_size_dict is None):
            print("Please insert either num_records or tables_size_dict")
            return()

        if cur_pipeline in [Pipeline.DescriptionToDB]:
            CodeTransformerObj = CodeTransformer(llm=self.llm)
            if self.code == '':
                CodeTransformerObj.generate_code_from_description(description=self.description, specifications=self.task_specifications,max_trials=10)
                self.code = CodeTransformerObj.code
            #generated_data = CodeTransformerObj.generate_data(table_size_dict=tables_size_dict, max_trials=3, output_format=output_format)
            full_query = compose_query_message(query=query, region=region, language=language)
            generated_data = CodeTransformerObj.generate_data(table_size_dict=tables_size_dict, max_trials=3, output_format=output_format, run_in_parallel=run_in_parallel, full_query=full_query)

        else:
            DataAugmentorObj = DataAugmentor(llm=self.llm, structure=self.data_structure_sample)
            if (examples_dataframe_dict is not None):
                #Curently supporting a single examples file. Needs to extend to support multi-tables
                first_key = list(examples_dataframe_dict.keys())[0]
                DataAugmentorObj.set_examples_dataframe(dataframe = examples_dataframe_dict[first_key],data_name = first_key)
            if run_in_parallel:
                generated_data = asyncio.run(DataAugmentorObj.generate_data_in_parallel(num_records=num_records, output_format=output_format, query=query, region=region, language=language))
            else:
                generated_data = DataAugmentorObj.generate_data(num_records=num_records, output_format=output_format, query=query, region=region, language=language)

        return generated_data

    def set_pipeline(self, pipelineName):
        self.pipeline_name = pipelineName