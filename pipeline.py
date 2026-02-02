from sagemaker import get_execution_role

from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.pipeline_context import PipelineSession

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

session = PipelineSession()

sklearn_processor = SKLearnProcessor(
    framework_version="0.20.0",
    role=get_execution_role(),
    instance_type="ml.t3.medium",
    instance_count=1,
    sagemaker_session=session
)

step_args = sklearn_processor.run(
    code="split_dataset.py",
    inputs=[
        ProcessingInput(source="shared/data.csv", destination="/opt/ml/processing/input")
    ],
    outputs=[
        ProcessingOutput(source="/opt/ml/processing/train"),
        ProcessingOutput(source="/opt/ml/processing/validation"),
        ProcessingOutput(source="/opt/ml/processing/test")
    ]
)

step_process = ProcessingStep(
    name="SplitDataset",
    processor=sklearn_processor,
    step_args=step_args
)

# ml.g4dn.xlarge