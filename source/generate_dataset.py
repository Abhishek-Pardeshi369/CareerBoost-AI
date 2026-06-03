import pandas as pd
import random
from utils import job_roles, all_skills

degrees = ["BCA", "BSc", "BTech", "MCA", "MSc"]

data = []

for _ in range(5000):

    role = random.choice(list(job_roles.keys()))
    required_skills = job_roles[role]

    row = {}

    row["JobRole"] = role
    row["Degree"] = random.choice(degrees)
    row["Experience"] = random.randint(0, 10)
    row["Projects"] = random.randint(0, 10)
    row["Certifications"] = random.randint(0, 5)
    row["Communication"] = random.randint(1, 10)

    matched = 0

    for skill in all_skills:

        if skill in required_skills:
            value = random.choices([1, 0], [85, 15])[0]
        else:
            value = random.choices([1, 0], [25, 75])[0]

        row[skill] = value

        if skill in required_skills and value == 1:
            matched += 1

    score = (
        matched * 10
        + row["Experience"] * 2
        + row["Projects"] * 2
        + row["Certifications"] * 3
        + row["Communication"]
    )

    row["Selected"] = 1 if score >= 50 else 0

    data.append(row)

df = pd.DataFrame(data)

df.to_csv("data/careerboost_dataset.csv", index=False)

print("Dataset Generated Successfully!")