"""
=========================================================
              CareerBoost AI
IT Career Fit Prediction & Guidance System
=========================================================

Developed By: Abhishek Pardeshi

Project Type:
Machine Learning Prediction Project with Streamlit UI

Project Objective:
To predict a candidate's suitability for various IT job roles
based on skills, education, experience, projects,
certifications, and communication score.

Key Features:
✓ Job Role Selection
✓ IT Skill Assessment
✓ Job Fit Prediction
✓ Skill Gap Analysis
✓ Career Recommendations
✓ Alternative Career Suggestions
✓ Interactive Streamlit Dashboard

Machine Learning Algorithm:
- Random Forest Classifier

Technologies Used:
- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Joblib

Target Users:
- Students
- Freshers
- Job Seekers
- Career Counselors

Version:
1.0

=========================================================
"""
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/careerboost_dataset.csv")

# Encode categorical columns
job_encoder = LabelEncoder()
degree_encoder = LabelEncoder()

df["JobRole"] = job_encoder.fit_transform(df["JobRole"])
df["Degree"] = degree_encoder.fit_transform(df["Degree"])

# Features & Target
X = df.drop("Selected", axis=1)
y = df["Selected"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy:.2f}")

# Save Model
joblib.dump(model, "models/model.pkl")
joblib.dump(job_encoder, "models/job_encoder.pkl")
joblib.dump(degree_encoder, "models/degree_encoder.pkl")

print("Model Saved Successfully!")
import streamlit as st
import pandas as pd
import joblib

from source.utils import job_roles, all_skills

# ------------------------
# Load Model
# ------------------------

model = joblib.load("models/model.pkl")
job_encoder = joblib.load("models/job_encoder.pkl")
degree_encoder = joblib.load("models/degree_encoder.pkl")

# ------------------------
# UI
# ------------------------

st.set_page_config(
    page_title="CareerBoost AI",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 CareerBoost AI")
st.subheader("IT Career Fit Prediction & Guidance System")

# ------------------------
# Inputs
# ------------------------

job_role = st.selectbox(
    "Select Job Role",
    list(job_roles.keys())
)

st.info(
    f"Required Skills for {job_role}: "
    + ", ".join(job_roles[job_role])
)

degree = st.selectbox(
    "Degree",
    ["BCA", "BSc", "BTech", "MCA", "MSc"]
)

experience = st.slider(
    "Experience (Years)",
    0,
    10,
    0
)

projects = st.slider(
    "Projects",
    0,
    10,
    0
)

certifications = st.slider(
    "Certifications",
    0,
    10,
    0
)

communication = st.slider(
    "Communication Score",
    1,
    10,
    5
)

st.markdown("## Select Your Skills")

selected_skills = []

cols = st.columns(4)

for i, skill in enumerate(all_skills):
    with cols[i % 4]:
        if st.checkbox(skill):
            selected_skills.append(skill)

# ------------------------
# Predict
# ------------------------

if st.button("Predict Job Fit"):

    row = {}

    row["JobRole"] = job_encoder.transform([job_role])[0]
    row["Degree"] = degree_encoder.transform([degree])[0]

    row["Experience"] = experience
    row["Projects"] = projects
    row["Certifications"] = certifications
    row["Communication"] = communication

    for skill in all_skills:
        row[skill] = 1 if skill in selected_skills else 0

    input_df = pd.DataFrame([row])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    score = int(probability * 100)

    st.markdown("---")

    st.subheader("Prediction Result")

    st.progress(score / 100)

    st.metric(
        "Job Fit Score",
        f"{score}%"
    )

    if prediction == 1:
        st.success("✅ Highly Suitable")
    else:
        st.error("❌ Not Recommended")

    # Missing Skills

    required = set(job_roles[job_role])
    user = set(selected_skills)

    missing = list(required - user)

    st.subheader("Missing Skills")

    if missing:
        for skill in missing:
            st.write(f"❌ {skill}")
    else:
        st.success("No Missing Skills")

    # Recommendations

    st.subheader("Recommendations")

    if missing:
        for skill in missing:
            st.write(f"📚 Learn {skill}")
    else:
        st.write("🎉 You match all required skills.")

    # Alternative Roles

    st.subheader("Alternative Career Options")

    alternatives = []

    for role, skills in job_roles.items():

        match_count = len(
            set(skills).intersection(user)
        )

        alternatives.append(
            (role, match_count)
        )

    alternatives.sort(
        key=lambda x: x[1],
        reverse=True
    )

    for role, match in alternatives[:3]:
        st.write(
            f"➡️ {role} ({match} skill matches)"
        )