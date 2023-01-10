# %%
import torch
import torchmetrics


def get_Accuracy(num_class):

    accuracy = torchmetrics.classification.BinaryAccuracy(
        threshold=0.5
    )

    return accuracy


def get_Precision(num_class):

    precision = torchmetrics.classification.BinaryPrecision(
    )

    return precision


def get_Dice():
    dice = torchmetrics.Dice(
        num_classes=2,
        average='micro',

    )
    return dice


# def get_Average_precision(num_class):

#     average_precision = torchmetrics.AveragePrecision(
#         # num_classes=num_class,
#         pos_label=0,
#         average=None
#     )

#     return average_precision


def get_Precision_Recall():
    precision_recall = torchmetrics.PrecisionRecallCurve(
        num_classes=1,
        pos_label=1
    )

    return precision_recall


def get_AUC():
    AUC = torchmetrics.AUC(
        reorder=True
    )
    return AUC


def get_F1Score():
    F1_Score = torchmetrics.F1Score(
        num_classes=2,

    )
    return F1_Score


def get_Confusion_Matrix():
    confusion_matrix = torchmetrics.classification.BinaryConfusionMatrix()

    return confusion_matrix

# %%

# preds = torch.tensor([1, 0.4, 0.8, 0.4])
# target = torch.tensor([1, 1, 0, 0])

# dice = get_Dice()
# accuracy = get_Accuracy(1)
# average_precision = get_Average_precision(1)
# precision = get_Precision(1)
# confusion_matrix = get_Confusion_Matrix()

# # result = accuracy(preds, target)
# acc = accuracy(preds, target)
# confusion  = confusion_matrix(preds, target)
# result = precision(preds, target)

# print(acc, result, confusion)
# %%
