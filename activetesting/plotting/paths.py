
def plus_base(base, end):
    return [base+e for e in end]


class ReproduciblePaths:
    """Store paths for experiment results."""
    base = 'outputs/final/'
    figure123 = [
        'SyntheticGPGP',
        'SyntheticQuadraticLinear',
        'SyntheticTwoMoonsRF']
    figure123 = plus_base(base, figure123)

    figure4 = [
        'SmallMNISTBNN',
        'SmallFMNISTResNet']

    figure4names = [
        'BNN MNIST',
        'ResNet-18 Fashion-MNIST']

    figure5 = base + 'LargeCIFAR100ResNet'
    figure5name = 'ResNet CIFAR100'

    figure6 = [
        'LargeFMNISTResNet',
        'LargeCIFAR10ResNet',
        'LargeCIFAR100WideResNet',
        'LargeCIFAR10ResNetAccuracy']
    figure6 = plus_base(base, figure6)
    figure6names = [
        'ResNet Fashion-MNIST',
        'ResNet CIFAR-10',
        'WideResNet CIFAR-100',
        'Resnet CIFAR-10 Accuracy']

    figure7 = base + 'LargeFMNISTBNN'

    old_figure4 = 'outputs/final/LargeMNISTBNN'


class LegacyPaths:
    """Legacy paths from before reproducible."""

    base = 'outputs/legacy/'
    figure123 = [
        '2020-12-31-12-00-30',
        '2020-12-30-19-18-12',
        '2021-01-06-20-27-51']
    figure123 = plus_base(base, figure123)

    figure4 = [
        [
            base+'2021-01-20-09-26-51',
            base+'2021-01-20-09-27-40',
            base+'2021-01-20-09-28-06',
            base+'2021-01-20-11-48-23',
        ],
        [
            base + '2021-01-20-11-40-35',
            base + '2021-01-24-14-57-58'
        ]
    ]
    figure4names = [
        'BNN MNIST',
        'ResNet-18 Fashion-MNIST']

    figure5 = base + '2021-01-20-16-33-33'
    figure5name = 'ResNet CIFAR100'

    figure6 = [
        [base + '2021-01-21-08-52-31'],
        [base + '2021-01-20-16-27-51'],
        [base + '2021-01-26-15-22-32'],
        [base + '2021-01-20-16-29-06']
    ]
    figure6names = [
        'ResNet Fashion-MNIST',
        'ResNet CIFAR-10',
        'WideResNet CIFAR-100',
        'Resnet CIFAR-10 Accuracy']

    figure7 = [
        '2021-01-20-10-12-35',
        '2021-01-20-10-17-31',
        '2021-01-22-08-41-17',
        '2021-01-22-08-43-13']
    figure7 = plus_base(base, figure7)

    old_figure4 = [
        '2021-01-20-09-53-03',
        '2021-01-20-10-10-36',
        '2021-01-22-07-32-21',
        '2021-01-22-07-33-23']
    old_figure4 = plus_base(base, old_figure4)

    base = 'outputs/legacy/'
    figureA1 = [
        '2020-12-31-12-00-30',
        '2020-12-30-22-34-23',
        '2020-12-31-11-00-13',
        '2020-12-30-19-18-12',
        '2021-01-06-20-27-51']

    figureA1 = plus_base(base, figureA1)


class OldLegacyPaths:
    """OldLegacy paths from before reproducible."""
    figure123 = [
        'outputs/keep/GPExperiment/2020-12-31-12-00-30',
        'outputs/keep/GPExperiment/2020-12-30-19-18-12',
        'outputs/keep/TwoMoonsExperiment/2021-01-06-20-27-51']

    base = 'outputs/outputs-azure/outputs/keep/MNISTExperiment/'
    figure4 = [
        [
            base+'2021-01-20-09-26-51',
            base+'2021-01-20-09-27-40',
            base+'2021-01-20-09-28-06',
            base+'2021-01-20-11-48-23',
        ],
        [
            'outputs/paper/2021-01-20-11-40-35',
            'outputs/paper/2021-01-24-14-57-58'
        ]
    ]

    figure4names = [
        'BNN MNIST',
        'ResNet-18 Fashion-MNIST']

    figure5 = 'outputs/paper/2021-01-20-16-33-33'
    figure5name = 'ResNet CIFAR100'

    figure6 = [
        ['outputs/paper/2021-01-21-08-52-31'],
        ['outputs/paper/2021-01-20-16-27-51'],
        ['outputs/paper/2021-01-26-15-22-32'],
        ['outputs/paper/2021-01-20-16-29-06']
    ]
    figure6names = [
        'ResNet Fashion-MNIST',
        'ResNet CIFAR-10',
        'WideResNet CIFAR-100',
        'Resnet CIFAR-10 Accuracy']

    base = 'outputs/outputs-azure/outputs/keep/MNISTExperiment/'
    figure7 = [
        '2021-01-20-10-12-35',
        '2021-01-20-10-17-31',
        '2021-01-22-08-41-17',
        '2021-01-22-08-43-13']
    figure7 = plus_base(base, figure7)

    base = 'outputs/outputs-azure/outputs/keep/MNISTExperiment/'
    old_figure4 = [
        '2021-01-20-09-53-03',
        '2021-01-20-10-10-36',
        '2021-01-22-07-32-21',
        '2021-01-22-07-33-23']
    old_figure4 = plus_base(base, old_figure4)
