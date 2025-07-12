# AI-Based-Medical-Diagnosis-Assistant
An intelligent medical diagnosis assistant that uses a trained Random Forest Classifier to predict possible diseases based on user-provided symptoms. It also provides treatment suggestions, loaded from a separate CSV file. Designed with a focus on data cleaning, robust error handling, and practical usage in health advisory systems.

### 🔍 Features
-  Machine learning model (Random Forest) trained on symptom-disease data
-  User inputs symptoms, receives top 3 likely diseases with confidence scores
-  Treatment suggestions for predicted diseases (from a separate CSV)
-  Smart warning system for unrecognized symptoms or missing data
-  Handles real-world issues like diseases with only 1 entry in training data
-  Model accuracy printed after training
-  Defensive coding: file checks, missing data handling, graceful exits
