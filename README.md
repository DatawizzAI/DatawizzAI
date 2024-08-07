# DatawizzAI: Synthetic Data Generation Toolkit

Welcome to DatawizzAI, an open-source Python package that leverages the power of Large Language Models (LLMs) to define, augment, and transform tabular and textual datasets. DatawizzAI is designed to simplify and accelerate the process of synthetic data creation, providing developers with a robust toolkit to innovate and expand the boundaries of data science.

## Features
- **Data Definition:** Easily define the structure and schema of your synthetic datasets.
- **Data Augmentation:** Expand your datasets by generating synthetic data that adheres to your defined structure.
- **Data Transformation:** Transform and enhance datasets by utilizing external knowledge for improved quality and insights.
- **Datasource Connectors:** Integrate with various data sources to extract data structures from existing datasets. (Coming Soon)
- **LLM Agnostic:** Connect to a variety of pre-trained LLMs for flexible data generation.

## Getting Started
1. **Installation:**
```python 
   pip install datawizzAI # Coming soon
```
2. **Quick Example:**
```python 
  from datawizzAI import DataGenerationPipeline, DataTransformer

  # Generate synthetic data
  DataGenerationPipelineObj = DataGenerationPipeline(llm = your_llm) # e.g. your_llm = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo")
  sample_synthetic_data = DataGenerationPipelineObj.extract_sample_data( description="Your data description here")
  full_synthetic_data = DataGenerationPipelineObj.generate_data(num_records = 30)

  # Transform your data
  DataTransformerObj = DataTransformer(llm=your_llm)
  sample_transformed_data = DataTransformerObj.define_transformation(source_data={"data-name": source_dataframe}, description="Your transformation description here")
  full_transformed_data = DataTransformerObj.transform(source_data=source_dataframe)

```
## Usage
Check out the [examples](https://github.com/DatawizzAI/DatawizzAI/blob/main/examples) for detailed usage examples and API references.

## Contributing
We welcome contributions from the community! Whether you're fixing a bug, adding a feature, or improving the docs, we appreciate your effort to make DatawizzAI better.

- **Contribute Code:** Please submit a pull request with a clear description of your changes.
- **Report Issues:** Use the issue tracker to report bugs or suggest enhancements.
- **Write Documentation:** Help new users get started by improving documentation.
See the [contributing guidelines](https://github.com/DatawizzAI/DatawizzAI/blob/main/CONTRIBUTING.md) for more information.

## Roadmap
- Add connectors for more data sources.
- Enhance LLM connectivity to include more pre-trained models.
- Optimize performance for large-scale data processing.

Note: Fine-tuning LLMs and support for vision/audio/multimodal data generation are currently out of the package's scope.

## Support
Need help? Join our [community forum](https://github.com/DatawizzAI/DatawizzAI/discussions) or reach out to us on [Slack](https://app.slack.com/client/T07446FN1QT/C0746PU91LL).

## License
DatawizzAI is open-sourced under the MIT License.

## Acknowledgments
Thanks to all the contributors who have helped shape DatawizzAI. A special thanks to the organizations and developers behind the LLMs that make this project possible.
