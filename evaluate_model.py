Python 3.12.0 (tags/v3.12.0:0fb18b0, Oct  2 2023, 13:03:39) [MSC v.1935 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> from robustbench.utils import load_model
... from robustbench.data import load_cifar10
... from robustbench.eval import clean_accuracy
... 
... # Load a robust model from RobustBench Zoo
... model = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf')
... model.eval()
... 
... # Load CIFAR-10 test dataset
... x_test, y_test = load_cifar10(n_examples=100)
... 
... # Evaluate the model's clean accuracy
... accuracy = clean_accuracy(model, x_test, y_test)
