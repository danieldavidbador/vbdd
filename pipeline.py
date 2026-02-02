import sagemaker

from sagemaker import get_execution_role

from sagemaker.image_uris import retrieve

from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline_context import PipelineSession

from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch, PyTorchModel

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

session = PipelineSession()

region = session.boto_region_name

training_image_uri = retrieve(
    framework="pytorch",
    region=region,
    version="2.6",
    py_version="py312",
    instance_type="ml.g4dn.xlarge",
    image_scope="training"
)

inference_image_uri = retrieve(
    framework="pytorch",
    region=region,
    version="2.6",
    py_version="py312",
    instance_type="ml.g4dn.xlarge",
    image_scope="inference",
)

input_data = ParameterString(
    name="InputData",
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
                source=input_data,
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

gan_estimator = PyTorch(
    entry_point="gan/train.py",
    role=get_execution_role(),
    image_uri=training_image_uri,
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
    image_uri=inference_image_uri,
    sagemaker_session=session,
)

create_gan_model = ModelStep(
    name="CreateGanModel",
    step_args=gan_model.create(
        instance_type="ml.m5.large",
        accelerator_type="ml.eia1.medium"
    )
)

register_gan_model = ModelStep(
    name="RegisterGanModel",
    step_args=gan_model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_name="Vibration Based GAN Model",
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
                source=train_gan_model.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/input/model"
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/output/train")
        ]
    )
)

detector_estimator = PyTorch(
    entry_point="detector/train.py",
    role=get_execution_role(),
    image_uri=training_image_uri,
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
    image_uri=inference_image_uri,
    sagemaker_session=session,
)

create_detector_model = ModelStep(
    name="CreateDetectorModel",
    step_args=detector_model.create(
        instance_type="ml.m5.large",
        accelerator_type="ml.eia1.medium"
    )
)

register_detector_model = ModelStep(
    name="RegisterDetectorModel",
    step_args=detector_model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_name="Vibration Based Damage Detection Model",
    )
)

pipeline = Pipeline(
    name = "VibrationBasedDamageDetectionModelPipeline",
    parameters = [input_data],
    steps = [
        split_dataset,
        train_gan_model,
        create_gan_model,
        register_gan_model,
        augment_dataset,
        train_detector_model,
        create_detector_model,
        register_detector_model
    ],
)

pipeline.upsert(role_arn=get_execution_role())

execution = pipeline.start(parameters=dict(InputData="s3://amazon-sagemaker-753565189289-us-east-1-bsrdcs7wf2k8iv/shared/data.csv"))

execution.describe()

execution.wait()

execution.list_steps()