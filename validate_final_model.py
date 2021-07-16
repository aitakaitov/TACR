import neural_net.validate_with_keywords
import neural_net.preprocessing
import neural_net.validation
from sys import argv

if len(argv) != 2:
    print("Specify the saved model to validate")
    exit()

model_path = "final_models/" + argv[1]
preprocessor = neural_net.preprocessing.Preprocessing().load("model_configurations/tokenizer_configurations/preprocessing-config-no-keyw")

# validate the model against the reference labels
print("Validating the model against true labels of FP dataset")
neural_net.validation.validate_model(model_path, preprocessor, "validation_datasets/validation_dataset_dev_fp",
                                     "validation_datasets/validation_dataset_dev_fp_annotated")

print("Validating the model against true labels of FP+TN dataset")
neural_net.validation.validate_model(model_path, preprocessor, "validation_datasets/validation_dataset_dev_extended",
                                     "validation_datasets/validation_dataset_dev_extended_annotated")

print("Validating the model against true labels of CONTROL dataset")
neural_net.validation.validate_model(model_path, preprocessor, "validation_datasets/validation_dataset_control",
                                     "validation_datasets/validation_dataset_control_annotated")

print("Validating the model in combination with keyword-based system on the CONTROL dataset")
neural_net.validate_with_keywords.validate_model_control(model_path, preprocessor)