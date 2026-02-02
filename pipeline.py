from sagemaker import get_execution_role

from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CreateModelStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline_context import PipelineSession

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorchModel

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

session = PipelineSession()

dataset_path = ParameterString(
    name="DatasetS3Path",
    default_value="s3://my-bucket/my-dataset/data.csv"
)

sklearn_processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=get_execution_role(),
    instance_type="ml.t3.medium",
    instance_count=1,
    sagemaker_session=session
)

split_dataset = ProcessingStep(
    name="SplitDataset",
    step_args=sklearn_processor.run(
        code="dataset/split.py",
        inputs=[
            ProcessingInput(
                source=dataset_path,
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test")
        ]
    )
)

gan_estimator = Estimator(
    entry_point="gan/train.py",
    role=get_execution_role(),
    framework_version="1.12.0",
    py_version="py38",
    instance_count=1,
    instance_type="ml.g4dn.xlarge",
    hyperparameters={
        "epochs": 10,
        "step-size": 5,
        "batch-size": 64,
        "learning-rate": 0.01,
    },
    sagemaker_session=session
)

train_gan_model = TrainingStep(
    name="TrainGanModel",
    step_args=gan_estimator.fit(
        inputs={
            "train": TrainingInput(
                s3_data=split_dataset.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv"
            )
        }
    )
)

gan_model = PyTorchModel(
    model_data=train_gan_model.properties.ModelArtifacts.S3ModelArtifacts,
    role=get_execution_role(),
    sagemaker_session=session,
)

create_gan_model = CreateModelStep(
    name="CreateGanModel",
    step_args=gan_model.create(
        instance_type="ml.m5.large",
        accelerator_type="ml.eia1.medium"
    )
)

augment_dataset = ProcessingStep(
    name="AugmentDataset",
    step_args=sklearn_processor.run(
        code="dataset/augment.py",
        inputs=[
            ProcessingInput(
                source=split_dataset.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                destination="/opt/ml/processing/input/data"
            ),
            ProcessingInput(
                source=create_gan_model.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/input/model"
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/output/train")
        ]
    )
)

detector_estimator = Estimator(
    entry_point="detector/train.py",
    role=get_execution_role(),
    framework_version="1.12.0",
    py_version="py38",
    instance_count=1,
    instance_type="ml.g4dn.xlarge",
    hyperparameters={
        "epochs": 15,
        "step-size": 5,
        "batch-size": 64,
        "learning-rate": 0.01,
    },
    sagemaker_session=session
)

train_detector_model = TrainingStep(
    name="TrainDetectorModel",
    step_args=detector_estimator.fit(
        inputs={
            "train": TrainingInput(
                s3_data=augment_dataset.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv"
            ),
            "validation": TrainingInput(
                s3_data=split_dataset.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv"
            )
        }
    )
)

detector_model = PyTorchModel(
    model_data=train_detector_model.properties.ModelArtifacts.S3ModelArtifacts,
    role=get_execution_role(),
    sagemaker_session=session,
)

create_detector_model = CreateModelStep(
    name="CreateDetectorModel",
    step_args=detector_model.create(
        instance_type="ml.m5.large",
        accelerator_type="ml.eia1.medium"
    )
)

pipeline = Pipeline(
    name = "VibrationBasedDamageDetectionModelPipeline",
    parameters = [dataset_path],
    steps = [split_dataset, train_gan_model, create_gan_model, augment_dataset, train_detector_model, create_detector_model],
)

pipeline.upsert(role_arn=get_execution_role())