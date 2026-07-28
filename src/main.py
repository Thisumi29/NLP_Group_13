from data_loading import load_and_merge_data
from data_preprocessing import preprocess_dataset
from eda import run_eda
from feature_engineering import (
    load_and_split_data,
    create_tfidf_features,
)
from random_forest_model import train_random_forest
from bilstm_model import train_bilstm


def run_feature_engineering():
    """Run train-test split and TF-IDF feature engineering."""

    (
        X_train_text,
        X_test_text,
        y_train,
        y_test,
    ) = load_and_split_data()

    create_tfidf_features(
        X_train_text,
        X_test_text,
    )


def show_menu():
    print("\nFake News Detection NLP Project")
    print("--------------------------------")
    print("1. Load and merge datasets")
    print("2. Preprocess dataset")
    print("3. Run EDA")
    print("4. Run feature engineering")
    print("5. Train Random Forest model")
    print("6. Train Bi-LSTM model")
    print("7. Run complete pipeline")
    print("0. Exit")


def run_complete_pipeline():
    """Run every project stage in order."""

    print("\nStep 1: Loading and merging data...")
    load_and_merge_data()

    print("\nStep 2: Preprocessing data...")
    preprocess_dataset()

    print("\nStep 3: Running EDA...")
    run_eda()

    print("\nStep 4: Training Random Forest...")
    train_random_forest()

    print("\nStep 5: Training Bi-LSTM...")
    train_bilstm()

    print("\nComplete NLP pipeline finished successfully.")


def main():
    while True:
        show_menu()
        choice = input("\nEnter your choice: ").strip()

        if choice == "1":
            load_and_merge_data()

        elif choice == "2":
            preprocess_dataset()

        elif choice == "3":
            run_eda()

        elif choice == "4":
            run_feature_engineering()

        elif choice == "5":
            train_random_forest()

        elif choice == "6":
            train_bilstm()

        elif choice == "7":
            run_complete_pipeline()

        elif choice == "0":
            print("Program closed.")
            break

        else:
            print("Invalid choice. Please enter a number from 0 to 7.")


if __name__ == "__main__":
    main()