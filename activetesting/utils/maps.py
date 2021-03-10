"""Map strings to classes."""
from activetesting.models import (
    LinearRegressionModel, GaussianProcessRegressor, RandomForestClassifier,
    SVMClassifier, GPClassifier, RadialBNN, TinyRadialBNN, ResNet18,
    WideResNet, ResNet18Ensemble, WideResNetEnsemble)
from activetesting.datasets import (
    QuadraticDatasetForLinReg, SinusoidalDatasetForLinReg,
    GPDatasetForGPReg, MNISTDataset, TwoMoonsDataset, FashionMNISTDataset,
    Cifar10Dataset, Cifar100Dataset)
from activetesting.acquisition import (
    RandomAcquisition, TrueLossAcquisition, DistanceBasedAcquisition,
    GPAcquisitionUncertainty,
    GPSurrogateAcquisitionLogLik, GPSurrogateAcquisitionMSE,
    ClassifierAcquisitionEntropy,
    RandomForestClassifierSurrogateAcquisitionEntropy,
    SVMClassifierSurrogateAcquisitionEntropy,
    GPClassifierSurrogateAcquisitionEntropy,
    RandomRandomForestClassifierSurrogateAcquisitionEntropy,
    GPSurrogateAcquisitionMSEDoublyUncertain,
    SelfSurrogateAcquisitionEntropy,
    BNNClassifierAcquisitionMI,
    AnySurrogateAcquisitionEntropy,
    ClassifierAcquisitionAccuracy,
    SelfSurrogateAcquisitionAccuracy,
    AnySurrogateAcquisitionAccuracy
    )
from activetesting.loss import (
    SELoss, MSELoss, RMSELoss, CrossEntropyLoss, AccuracyLoss)

from activetesting.risk_estimators import (
    BiasedRiskEstimator, NaiveUnbiasedRiskEstimator,
    FancyUnbiasedRiskEstimator, TrueRiskEstimator,
    ImportanceWeightedRiskEstimator, TrueUnseenRiskEstimator,
    FancyUnbiasedRiskEstimatorCut, FancyUnbiasedRiskEstimatorCut1,
    FancyUnbiasedRiskEstimatorCut2,
    FancyUnbiasedRiskEstimatorCut3
    )

model = dict(
    LinearRegressionModel=LinearRegressionModel,
    GaussianProcessRegressor=GaussianProcessRegressor,
    RandomForestClassifier=RandomForestClassifier,
    SVMClassifier=SVMClassifier,
    GPClassifier=GPClassifier,
    RadialBNN=RadialBNN,
    TinyRadialBNN=TinyRadialBNN,
    ResNet18=ResNet18,
    WideResNet=WideResNet,
    ResNet18Ensemble=ResNet18Ensemble,
    WideResNetEnsemble=WideResNetEnsemble,
)

dataset = dict(
    QuadraticDatasetForLinReg=QuadraticDatasetForLinReg,
    SinusoidalDatasetForLinReg=SinusoidalDatasetForLinReg,
    GPDatasetForGPReg=GPDatasetForGPReg,
    MNISTDataset=MNISTDataset,
    TwoMoonsDataset=TwoMoonsDataset,
    FashionMNISTDataset=FashionMNISTDataset,
    Cifar10Dataset=Cifar10Dataset,
    Cifar100Dataset=Cifar100Dataset,
)

acquisition = dict(
    RandomAcquisition=RandomAcquisition,
    TrueLossAcquisition=TrueLossAcquisition,
    DistanceBasedAcquisition=DistanceBasedAcquisition,
    GPAcquisitionUncertainty=GPAcquisitionUncertainty,
    GPSurrogateAcquisitionLogLik=GPSurrogateAcquisitionLogLik,
    GPSurrogateAcquisitionMSE=GPSurrogateAcquisitionMSE,
    ClassifierAcquisitionEntropy=ClassifierAcquisitionEntropy,
    RandomForestClassifierSurrogateAcquisitionEntropy=(
        RandomForestClassifierSurrogateAcquisitionEntropy),
    SVMClassifierSurrogateAcquisitionEntropy=(
        SVMClassifierSurrogateAcquisitionEntropy),
    GPClassifierSurrogateAcquisitionEntropy=(
        GPClassifierSurrogateAcquisitionEntropy),
    RandomRandomForestClassifierSurrogateAcquisitionEntropy=(
        RandomRandomForestClassifierSurrogateAcquisitionEntropy),
    GPSurrogateAcquisitionMSEDoublyUncertain=(
        GPSurrogateAcquisitionMSEDoublyUncertain),
    SelfSurrogateAcquisitionEntropy=SelfSurrogateAcquisitionEntropy,
    BNNClassifierAcquisitionMI=BNNClassifierAcquisitionMI,
    AnySurrogateAcquisitionEntropy=AnySurrogateAcquisitionEntropy,
    ClassifierAcquisitionAccuracy=ClassifierAcquisitionAccuracy,
    SelfSurrogateAcquisitionAccuracy=SelfSurrogateAcquisitionAccuracy,
    AnySurrogateAcquisitionAccuracy=AnySurrogateAcquisitionAccuracy,
)

loss = dict(
    SELoss=SELoss,
    MSELoss=MSELoss,
    RMSELoss=RMSELoss,
    CrossEntropyLoss=CrossEntropyLoss,
    AccuracyLoss=AccuracyLoss,
)

risk_estimator = dict(
    TrueRiskEstimator=TrueRiskEstimator,
    BiasedRiskEstimator=BiasedRiskEstimator,
    NaiveUnbiasedRiskEstimator=NaiveUnbiasedRiskEstimator,
    FancyUnbiasedRiskEstimator=FancyUnbiasedRiskEstimator,
    ImportanceWeightedRiskEstimator=ImportanceWeightedRiskEstimator,
    TrueUnseenRiskEstimator=TrueUnseenRiskEstimator,
    FancyUnbiasedRiskEstimatorCut=FancyUnbiasedRiskEstimatorCut,
    FancyUnbiasedRiskEstimatorCut1=FancyUnbiasedRiskEstimatorCut1,
    FancyUnbiasedRiskEstimatorCut2=FancyUnbiasedRiskEstimatorCut2,
    FancyUnbiasedRiskEstimatorCut3=FancyUnbiasedRiskEstimatorCut3
)
