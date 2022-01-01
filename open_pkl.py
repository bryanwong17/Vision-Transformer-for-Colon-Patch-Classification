import pickle
import os

def main():
    train_loss_list = []
    valid_loss_list = []
    train_acc_list = []
    valid_acc_list = []

    # for filename in os.listdir("sample_data_colon/vit_base_16/big_data/pretrained/log"):
    with open("sample_data_colon/vit_base_16/big_data/pretrained/log/log_1.pkl", 'rb') as f:
        content = pickle.load(f)
        print(content)
        # train_loss_list.append(content["train_loss"])
        # valid_loss_list.append(content["valid_loss"])
        # train_acc_list.append(content["train_acc"])
        # valid_acc_list.append(content["valid_acc"])

if __name__ == "__main__":
    main()