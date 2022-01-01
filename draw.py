import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from spicy import interp
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix

# from dataset import DATA_PATH_CASSAVA
# from train import DATA_PATH_VIT_BASE_16

# DATA_PATH_CASSAVA = "cassava-leaf-disease-classification"
# DATA_PATH_VIT_BASE_16_224 = "sample_data_colon/vit_base_16_224"
# DATA_PATH_VIT_BASE_16_384 = "sample_data_colon/vit_base_16_384"
# DATA_PATH_VIT_BASE_32_384 = "sample_data_colon/vit_base_32_384"
# DATA_PATH_VIT_BASE_16 = "sample_data_colon/vit_base_16"
# DATA_PATH_DENSENET201 = "sample_data_colon/densenet201"

def get_roc_curve(save_path, label_actual, predicted_score):
    fpr = {}
    tpr = {}
    roc_auc = {}

    n_classes = 3

    label_actual = np.array(label_actual)
    predicted_score = np.array(predicted_score)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(label_actual, predicted_score, pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.plot(fpr[0], tpr[0], linestyle="--", color="aqua", label="ROC Curve of Normal Class area = " + str(roc_auc[0]))
    plt.plot(fpr[1], tpr[1], linestyle="--", color="darkorange", label="ROC Curve of Dysplasia Class area = " + str(roc_auc[1]))
    plt.plot(fpr[2], tpr[2], linestyle="--", color="cornflowerblue", label="ROC Curve of Malignant Class area = " + str(roc_auc[2]))
    plt.plot([0, 1], [0, 1], 'k--')
    # plt.grid()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Colon Patch Classification ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='lower right')
    plt.savefig(save_path + "/roc_curve.png")
    plt.close()

def get_confusion_matrix(save_path, label_actual, label_predicted):
    cm = confusion_matrix(label_actual, label_predicted, labels=[0,1,2])
    print(cm)
    cm_df = pd.DataFrame(cm, index=["Normal", "Dysplasia", "Malignant"], columns=["Normal", "Dysplasia", "Malignant"])
    # plt.figure(figsize=(10,8))
    sns.heatmap(cm_df, annot=True, cmap="Blues", fmt=".1f")
    plt.title("Confusion Matrix for Colon Patch Classification")
    plt.ylabel("Predicted Classes")
    plt.xlabel("Actual Classes")
    plt.savefig(save_path + "/confusion_matrix.png")
    plt.close()

def get_loss_curve(save_path, train_losses, valid_losses):
    plt.plot(train_losses, color="blue", label="train")
    plt.plot(valid_losses, color="red", label="valid")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path + "/lose_curve.png")
    plt.close()

def get_accuracy_curve(save_path, train_acc, valid_acc):
    plt.plot(train_acc, color="blue", label="train")
    plt.plot(valid_acc, color="red", label="valid")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path + "/accuracy_curve.png")
    plt.close()

if __name__ == "__main__":
    pass

