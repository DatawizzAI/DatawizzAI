class Pipeline:
    """
    A set of functionalities for mapping the supported pipelines to their respective names.
    """

    DescriptionToMLDataset = 'DescriptionToMLDataset'
    DescriptionToDB = 'DescriptionToDB'
    SQLToTabular = 'SQLToTabular'
    DescriptionToUnstructured = 'DescriptionToUnstructured'
    ExamplesDataframeToTabular = 'ExamplesDataframeToTabular'
    APISpecificationToData = 'APISpecificationToData'
    UNKNOWN = 'UNKNOWN'

    pipelines_dict = {
        DescriptionToMLDataset: """Generate a single flattened table with synthetic records that can be used for training a well-performing ML model for the task described in the data description.\
            Please include any relevant attribute you can think of, as we do not want to miss any feature that the user may find useful.\
            Format the output as JSON with the data name as key, and the records as items.""",
        DescriptionToDB: """Generate Sample synthetic records for relational DB (one or more tables).\
            Please follow any relevant distributions for the stated fields, as we want this data to be as valid and useful as possible for development and testing.\
            Format the output as JSON with each table name as key and in the nested level the feature names as keys.""",
        SQLToTabular: """Generate Sample synthetic data that fit to the structure given by the SQL command that appears in the data description.\
            Please follow any relevant distributions for the stated fields, as we want this data to be as valid and useful as possible for development and testing.\
            Format the output as JSON with each table name as key.""",
        DescriptionToUnstructured: """Generate Sample data, potentially unstructured, that fit to the data description.\
            Format the output as JSONL with the dataset name as key and each line contains a single sampled text in the appropriate format.""",
        ExamplesDataframeToTabular: """Generate Sample data that fit to the example structure given in the data description.\
            Format the output as a JSONL file, where each line contains a single sampled text.""",
        APISpecificationToData: """Check the input and output of the API that is mentioned in the user desription, and generate synthetic API calls (with valid input and output) that fit to the required structure of this API.\
            Format your output as JSONL file, with the actual API name as key and each line containing the input object and the output object of a synthetic API call.""",
        UNKNOWN: '',
    }

    pipelines_descriptions = """
    DescriptionToMLDataset: Generate Sample synthetic records for tabular data (preferably a single flattened table) that can be used for training a well-performing ML model for the task described in the data description.\
            Please include any relevant attribute you can think of, as we do not want to miss any feature that the user may find useful.\
            Format the output as JSON with the dataset name as key.
    DescriptionToDB: Generate Sample synthetic records for relational DB (one or more tables).\
            Please follow any relevant distributions for the stated fields, as we want this data to be as valid and useful as possible for development and testing.\
            Format the output as JSON with each table name as key and in the nested level the feature names as keys.
    SQLToTabular: Generate Sample synthetic data that fit to the structure given by the SQL command that appears in the data description.\
            Please follow any relevant distributions for the stated fields, as we want this data to be as valid and useful as possible for development and testing.\
            Format the output as JSON with each table name as key.
    DescriptionToUnstructured: Generate Sample data, potentially unstructured, that fit to the data description.\
            Format the output as JSONL with the dataset name as key and each line contains a single sampled text in the appropriate format.
    ExamplesDataframeToTabular: Generate Sample data that fit to the example structure given in the data description.\
            Format the output as a JSONL file, where each line contains a single sampled text.
    APISpecificationToData: Check the input and output of the API that is mentioned in the user desription, and generate synthetic API calls (with valid input and output) that fit to the required structure of this API.
            Format your output as JSONL file, with the actual API name as key and each line containing the input object and the output object of a synthetic API call.
    UNKNOWN: ''
    """


    def get_pipeline_prompt(pipeline_name):
        """
        Get the prompt template corresponding to a given pipeline name.

        Parameters:
        - pipeline_name: The pipeline name.

        Returns:
        str: The prompt template for the specified pipeline.
        """
        return Pipeline.pipelines_dict.get(pipeline_name, '')


    def get_pipeline_name(label):
        pipeline_mapping = {
            'DescriptionToMLDataset': Pipeline.DescriptionToMLDataset,
            'DescriptionToDB': Pipeline.DescriptionToDB,
            'SQLToTabular': Pipeline.SQLToTabular,
            'DescriptionToUnstructured': Pipeline.DescriptionToUnstructured,
            'ExamplesDataframeToTabular': Pipeline.ExamplesDataframeToTabular,
            'APISpecificationToData': Pipeline.APISpecificationToData,
            'UNKNOWN': Pipeline.UNKNOWN
        }

        if label in pipeline_mapping:
            return pipeline_mapping[label]
        else:
            raise ValueError("Invalid pipeline name")