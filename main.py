
from data_preprocessing import dataclean_preprocessing,preparing_traindataset
from model_build import model_build
from model_train import model_train

def main():
    train_data, val_data, test_data = dataclean_preprocessing('Data/apple.csv')
    X_train, y_train, X_val, y_val, X_test, y_test = preparing_traindataset(train_data, val_data, test_data)
    model_new = model_build()
    model_train(model_new, X_train, y_train, X_val, y_val, 5)


if __name__ == '__main__':
    main()

