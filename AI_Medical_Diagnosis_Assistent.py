import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# --- FILE PATHS ---
KAGGLE_TRAINING_CSV = 'Training.csv'
YOUR_ORIGINAL_DISEASES_SYMPTOMS_CSV = 'Diseases_Symptoms.csv'

treatment_dict = {}

# ---------- Load and Preprocess Dataset ----------
try:
    df_model_data = pd.read_csv(KAGGLE_TRAINING_CSV)

    if 'prognosis' in df_model_data.columns:
        df_model_data.rename(columns={'prognosis': 'name'}, inplace=True)
    else:
        raise ValueError(f"The dataset '{KAGGLE_TRAINING_CSV}' must contain a 'prognosis' column for disease names.")

    df_model_data.columns = [col.lower().strip() for col in df_model_data.columns]

    if 'name' not in df_model_data.columns:
        raise ValueError(f"The 'name' column (formerly 'prognosis') was not found or incorrectly processed in '{KAGGLE_TRAINING_CSV}'.")

    df_original_treatments = pd.read_csv(YOUR_ORIGINAL_DISEASES_SYMPTOMS_CSV)

    if 'Code' in df_original_treatments.columns:
        df_original_treatments.rename(columns={'Code': 'Name'}, inplace=True)

    else:

        if 'Name' not in df_original_treatments.columns:
            raise ValueError(f"The treatment dataset '{YOUR_ORIGINAL_DISEASES_SYMPTOMS_CSV}' must contain either a 'Code' or 'Name' column for diseases.")

    if 'Treatments' not in df_original_treatments.columns:
        raise ValueError(f"The treatment dataset '{YOUR_ORIGINAL_DISEASES_SYMPTOMS_CSV}' must contain a 'Treatments' column.")

    df_original_treatments['Name'] = df_original_treatments['Name'].astype(str).str.lower()
    df_original_treatments['Treatments'] = df_original_treatments['Treatments'].fillna('').astype(str).str.lower()


    treatment_dict = dict(zip(df_original_treatments['Name'], df_original_treatments['Treatments']))


except FileNotFoundError as e:
    print(f"Error: Required CSV file not found: {e}. Make sure '{KAGGLE_TRAINING_CSV}' and '{YOUR_ORIGINAL_DISEASES_SYMPTOMS_CSV}' are in the same directory (or correct Colab path).")
    exit()
except ValueError as e:
    print(f"Error processing dataset: {e}")
    print(f"Please check your CSV files for correct column names and structure.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during dataset loading: {e}")
    exit()

X = df_model_data.drop(columns=['name'])
y = df_model_data['name']

all_symptoms = list(X.columns)

X.fillna(0, inplace=True)

disease_counts = y.value_counts()
single_entry_diseases = disease_counts[disease_counts < 2].index.tolist()

if single_entry_diseases:
    print("\nWARNING: The following diseases have only ONE entry in the Kaggle dataset used for training:")
    for disease in single_entry_diseases:
        print(f"- {disease}")
    print("These will be filtered out to ensure stratify=y works, but the model cannot learn about them from only one example.")
    print("Consider adding more entries for these diseases in your Kaggle dataset if they are critical, or they will be ignored by the model.")

    indices_to_keep = y[~y.isin(single_entry_diseases)].index
    X = X.loc[indices_to_keep]
    y = y.loc[indices_to_keep]

    if X.empty or y.empty:
        print("Error: After filtering single-entry diseases from Kaggle data, no data remains. Cannot train model.")
        exit()

if X.empty or y.empty:
    print("Error: No data or labels found after preprocessing. Check your CSVs and filtering steps.")
    exit()

if X.shape[1] == 0:
    print("Error: No symptom columns found in the Kaggle dataset after processing.")
    exit()

# ---------- Train Model ----------
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')

n_classes = len(y.unique())
total_samples = len(X)

min_samples_for_stratification = n_classes * 2

if total_samples < min_samples_for_stratification:
    test_size_param = max(2, int(total_samples * 0.2))
    if total_samples - test_size_param < n_classes:
        test_size_param = max(2, total_samples - n_classes)
        if test_size_param < 1: test_size_param = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_param, random_state=42)
else:
    target_test_size_fraction = 0.2
    calculated_test_samples = max(n_classes, int(total_samples * target_test_size_fraction))

    if total_samples - calculated_test_samples < n_classes:
        calculated_test_samples = total_samples - n_classes

    test_size_param = calculated_test_samples

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_param, random_state=42, stratify=y)


if X_train.empty or y_train.empty:
    print("Error: Training data is empty after split. This usually means your dataset is too small or filtered too aggressively.")
    exit()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ---------- Diagnosis Assistant Function ----------
def diagnose_patient(name, age, gender, symptoms_input):
    symptoms_input = [s.strip().lower() for s in symptoms_input]

    input_vector_dict = {symptom_col: 0 for symptom_col in all_symptoms}
    for symptom in symptoms_input:
        if symptom in input_vector_dict:
            input_vector_dict[symptom] = 1
        else:
            print(f"Warning: Symptom '{symptom}' not recognized by the model. It will be ignored.")

    input_vector = [input_vector_dict[symptom_col] for symptom_col in all_symptoms]

    if not any(input_vector):
        predicted_disease = "No specific disease could be identified based on the symptoms provided or known symptoms."
        treatment_plan = "Please consult a medical professional for a proper diagnosis if symptoms persist or worsen. Consider providing more detailed symptoms."
    else:
        try:
            probabilities = model.predict_proba([input_vector])[0]
            disease_probabilities = list(zip(model.classes_, probabilities))
            disease_probabilities.sort(key=lambda x: x[1], reverse=True)

            predicted_disease = disease_probabilities[0][0]
            predicted_probability = disease_probabilities[0][1]

            common_mild_symptoms = {
                'fever', 'headache', 'cough', 'sore_throat', 'runny_nose', 'fatigue',
                'body_aches', 'sneezing', 'dizziness', 'chills', 'mild_fever', 'sweating',
                'vomiting', 'diarrhoea', 'abdominal_pain', 'irritability'
            }

            severe_diseases = {
                "aids", "paralysis (brain hemorrhage)", "jaundice", "malaria", "dengue",
                "typhoid", "hepatitis b", "hepatitis c", "hepatitis d", "pneumonia",
                "peptic ulcer diseae", "hypertension", "diabetes ", "bronchial asthma",
                "cervical spondylosis", "tuberculosis", "heart attack", "stroke",
                "chicken pox", "arthritis", "migraine", "tuberculosis", "encephalitis",
                "lymphoma", "leukemia", "adrenal cancer", "breast cancer", "testicular cancer",
                "endometrial cancer", "esophageal cancer", "liver cancer", "bone cancer"
            }

            input_symptoms_set = set(symptoms_input)
            is_predominantly_mild = len(input_symptoms_set.intersection(common_mild_symptoms)) >= (len(input_symptoms_set) * 0.8)

            if (is_predominantly_mild and
                predicted_disease.lower() in severe_diseases and
                predicted_probability < 0.5):

                predicted_disease = "Symptoms suggest a common mild illness. Please monitor closely."
                treatment_plan = "Rest, stay hydrated, and consider over-the-counter medications for symptom relief. If symptoms worsen or persist, or if new severe symptoms appear, please consult a medical professional immediately. This AI is not a substitute for professional medical advice."
            else:
                treatment_plan = treatment_dict.get(predicted_disease.lower(),
                                                     "Consult a medical professional for proper treatment. This AI is for informational purposes only and not a substitute for professional medical advice.")

            print("\nTop 3 possible diagnoses:")
            for disease, prob in disease_probabilities[:3]:
                print(f"   - {disease} (Confidence: {prob:.2f})")
        except Exception as e:
            predicted_disease = "An error occurred during prediction."
            treatment_plan = f"Unable to predict due to model or data issue: {e}. Please ensure the dataset is sufficient and the model is trained."

    print("\n" + "="*50)
    print("       AI-Based Medical Diagnosis Report")
    print("="*50)
    print(f"👤 Name       : {name}")
    print(f"🎂 Age        : {age}")
    print(f"⚥ Gender      : {gender}")
    print(f"📝 Symptoms   : {', '.join(symptoms_input)}")
    print("-" * 50)
    print(f"🔍 Predicted Disease : {predicted_disease}")
    print(f"💊 Suggestion: {treatment_plan}")
    print("="*50 + "\n")

# ---------- Run ----------
if __name__ == "__main__":
    print("-" * 50)
    print("Welcome to the AI-Based Medical Diagnosis Assistant!")
    print("-" * 50)
    name = input("Enter your name: ")
    age = input("Enter your age: ")
    gender = input("Enter your gender (e.g., Male, Female, Other): ")

    print("\nEnter symptoms separated by commas (e.g., itching,skin_rash,continuous_sneezing):")
    symptoms_raw = input("Symptoms: ")
    symptoms_list = [s.strip().lower() for s in symptoms_raw.split(',')]

    diagnose_patient(name, age, gender, symptoms_list)