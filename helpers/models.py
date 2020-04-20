from enum import Enum


class ModelTypeEnum(Enum):
    RESNET = 0
    SQEEZENET = 1
    INCEPTIONV3 = 2
    DENSENET = 3


class ModelMixin:
    MODEL_TYPE_SET_LOOKUP = {
        ModelTypeEnum.RESNET: lambda x: x.setModelTypeAsResNet(),
        ModelTypeEnum.SQEEZENET: lambda x: x.setModelTypeAsSqueezeNet(),
        ModelTypeEnum.INCEPTIONV3: lambda x: x.setModelTypeAsInceptionV3(),
        ModelTypeEnum.DENSENET: lambda x: x.setModelTypeAsDenseNet(),
    }

    def _set_proper_model_type(self, model_type, trainer_or_detector):
        self.MODEL_TYPE_SET_LOOKUP[model_type](trainer_or_detector)
