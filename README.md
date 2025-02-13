# AutoAttacks_ART
Implementation and Evaluation of Adversarial Robustness Attacks using Adversarial Robustness Toolbox(ART)


The project employs a combination of traditional and advanced methodologies in machine learning and adversarial robustness research to evaluate and improve the resilience of models against adversarial attacks. The primary methodology used in this project is the implementation of adversarial attacks on machine learning models to test their robustness. Several attack types are used, including:

•	Evasion Attacks: These attacks focus on manipulating input data during inference to deceive the model into making incorrect predictions. Common evasion attacks include Fast Gradient Sign Method (FGSM), Basic Iterative Method (BIM), and Carlini & Wagner attack. These attacks work by adding perturbations to the input data that are imperceptible to humans but cause the model to misclassify the input.

•	Poisoning Attacks: This attack targets the model during training by injecting harmful data into the training set, influencing the model's learning process. Backdoor Attacks are commonly used in poisoning scenarios, where a subset of the data is manipulated to introduce hidden vulnerabilities in the trained model.

•	Model Extraction Attacks: This involves extracting a surrogate model from a target model by querying the model and observing its responses. These attacks aim to recreate a similar model by reverse-engineering the target’s behavior, often with the goal of stealing intellectual property or probing for vulnerabilities.

•	Membership Inference Attacks: These attacks aim to infer whether a particular data point was used in the model’s training set. The methodology typically involves querying the model and observing the behavior of its output to determine membership.

These attacks are implemented using the Adversarial Robustness Toolbox (ART), which provides pre-built attack algorithms for various types of adversarial threats. By leveraging ART's modular attack functionality, the project efficiently tests a range of attacks on machine learning models.


(USE execution_commands.txt TO EXECUTE EACH ATTACK)

