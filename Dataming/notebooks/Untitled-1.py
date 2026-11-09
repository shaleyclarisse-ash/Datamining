# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

# %%
df = pd.read_csv("../data/Student_perf.csv")

df.shape

# %%
print("Dataset shape:", df.shape)
print("Duplicates:", df.duplicated().sum())

# %%
print("Missing Values:")
print(df.isna().sum())

# %%
df.info()

# %%
df.describe()

# %%
print(df.columns.tolist())

# %%
print(df.shape)
print(df.columns.tolist())
print(df.isna().sum())

# %%
missing_rows  = df[df.isna().any(axis=1)]

print(missing_rows.to_string(index=False))

# %%
print("\nDuplicate rows:", df.duplicated().sum())

# %%
print("\nRisk group counts:")
print(df["Risk_Label"].value_counts())

# %%
print("\nRisk group percentages:")
print(
    df["Risk_Label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# %%
clean = df.copy()

clean["Study_Hours_Per_Week"] = (
    clean["Study_Hours_Per_Week"]
    .fillna(clean["Study_Hours_Per_Week"].median())
)

# %%
columns_to_compare = [
    "Attendance_Pct",
    "Study_Hours_Per_Week",
    "Assignment_Avg",
    "Previous_Exam_Score",
    "Final_Score"
]

risk_comparison = (
    df.groupby("Risk_Label")[columns_to_compare]
    .mean()
    .round(2)
)

print(risk_comparison)

# %%
import matplotlib.pyplot as plt

final_scores = df.groupby("Risk_Label")["Final_Score"].mean()

final_scores.plot(kind="bar", figsize=(7,4))

plt.title("Average Final Score by Risk Group")
plt.xlabel("Risk Group")
plt.ylabel("Average Final Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

#The Not At Risk group had an average final score of 81.70, compared
#with 59.56 for At Risk group, a difference of 22.14 points.

# %%
attendance = df.groupby("Risk_Label")["Attendance_Pct"].mean()

attendance.plot(kind="bar", figsize=(7, 4))

plt.title("Average Attendance Risk Group")
plt.xlabel("Risk Group")
plt.ylabel("Average Attendance (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

#Average attendance was 90.82% for the Not At Risk group and 70.22% for
#the At Risk group, a difference of 20.60 percentange points

# %%
numeric_cols = [
    "Attendance_Pct",
    "Study_Hours_Per_Week",
    "Assignment_Avg",
    "Previous_Exam_Score",
    "Final_Score"
]

correlations = (
    df[numeric_cols]
    .corr()["Final_Score"]
    .drop("Final_Score")
    .sort_values(ascending=False)
)

print(correlations.round(3))

# %%
clean = df.copy()

print("Missing values before cleaning:")
print(clean.isna().sum())

duplicates = df.duplicated().sum
print("\nExact duplicate rows:", duplicates)

clean = clean.drop_duplicates().copy()

numeric_cols = [
    "Attendance_Pct",
    "Study_Hours_Per_Week",
    "Assignment_Avg",
    "Previous_Exam_Score"
]

for col in numeric_cols:
    clean[col] = clean[col].fillna(clean[col].median())
    
    print("\nMissing values after cleaning:")
    print(clean.isna().sum())
    
    print("\nFinal dataset shape:", clean.shape)

# %%
program_scores = (
    clean.groupby("Program")["Final_Score"]
    .mean()
    .sort_values(ascending=False)
    .round(2)
)

print("Average Final Score by Program:")
print(program_scores)

# %%
program_scores.plot(kind="bar", figsize=(7, 4))

import matplotlib.pyplot as plt

plt.title("Average Final Score by Program")
plt.xlabel("Academic Program")
plt.ylabel("Average Final Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

#BSIT recorded the highest average final score (75.75), followed by BSCS (69.60) and BSIS (58.00). 
#The 17.75-point difference between BSIT and BSIS suggests that it would be useful to investigate 
# whether academic support needs or other relevant factors differ across programs.

# %%
year_scores = (
    clean.groupby("Year_Level")["Final_Score"]
    .mean()
    .sort_values(ascending=False)
    .round(2)
)

print("Average Final Score by Year Level:")
print(year_scores)

# %%
year_scores.plot(kind="bar", figsize=(7, 4))

import matplotlib.pyplot as plt

plt.title("Average Final Score by Year Level")
plt.xlabel("Year Level")
plt.ylabel("Average Final Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# %%
risk_by_program = pd.crosstab(
    clean["Program"],
    clean["Risk_Label"],
    normalize="index"
) * 100

print("Risk percentage by program:")
print(risk_by_program.round(2))

# %%
import matplotlib.pyplot as plt

risk_by_program.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5)
)

plt.title("Academic Risk Percentage by Program")
plt.xlabel("Program")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=0)
plt.legend(title="Risk Label")
plt.tight_layout()
plt.show()

# %%
numeric_data = clean.select_dtypes(include="number")

statistics = numeric_data.agg([
    "mean",
    "median",
    "std",
    "min",
    "max"
]).T

print("Descriptive Statistics")
print(statistics.round(2).to_string())

# %%
# Find invalid attendance percentages
invalid_attendance = clean[
    (clean["Attendance_Pct"] < 0) |
    (clean["Attendance_Pct"] > 100)
]

print("INVALID ATTENDANCE RECORDS")
print(invalid_attendance.to_string(index=False))

print("Total invalid records:", len(invalid_attendance))


# %%
clean.loc[
    (clean["Attendance_Pct"] < 0) |
    (clean["Attendance_Pct"] > 100),
    "Attendance_Pct"
] = float("nan")

valid_median = clean["Attendance_Pct"].median()

clean["Attendance_Pct"] = clean["Attendance_Pct"].fillna(valid_median)

print("Median used:", valid_median)
print("New maximum:", clean["Attendance_Pct"].max())

print("Remaining invalid attendance:",
      ((clean["Attendance_Pct"] < 0) |
       (clean["Attendance_Pct"] > 100)).sum())


# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(9, 5))

sns.histplot(
    data=clean,
    x="Final_Score",
    bins=10,
    kde=True
)

plt.title("Distribution of Students' Final Scores")
plt.xlabel("Final Score")
plt.ylabel("Number of Students")
plt.tight_layout()

plt.savefig("final_score_distribution.png", dpi=300)
plt.show()


# %%
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.barplot(
    data=clean,
    x="Program",
    y="Final_Score",
    order=["BSCS", "BSIS", "BSIT"],
    errorbar=None,
    ax=axes[0]
)

axes[0].set_title("Average Final Score by Program")
axes[0].set_xlabel("Program")
axes[0].set_ylabel("Average Final Score")
axes[0].set_ylim(1,100)


sns.barplot(
    data=clean,
    x="Year_Level",
    y="Final_Score",
    order=["1st", "2nd", "3rd", "4th"],
    errorbar=None,
    ax=axes[1]
)

axes[1].set_title("Average Final Score by Year Level")
axes[1].set_xlabel("Year Level")
axes[1].set_ylabel("Average Final Score")
axes[1].set_ylim(0, 100)

plt.tight_layout
plt.savefig("average_final_scores.png", dpi=300)
plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(9, 5))

sns.regplot(
    data=clean,
    x="Attendance_Pct",
    y="Final_Score",
    scatter_kws={"alpha": 0.7 },
    line_kws={"color": "red"}
)

plt.title("Relationship Between Attendance and Final Score")
plt.xlabel("Attendance Percentage (%)")
plt.ylabel("Final Score")

plt.xlim(0, 105)
plt.ylim(0, 105)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("attendance_vs_final_score.png", dpi=300)
plt.show()

# %%
predictors = [
    "Attendance_Pct",
    "Study_Hours_Per_Week",
    "Assignment_Avg",
    "Previous_Exam_Score"
]

correlations = clean[
    predictors + ["Final_Score"]
].corr()["Final_Score"].drop("Final_Score")

print("CORRELATION WITH FINAL SCORE")
print(correlations.sort_values(ascending=False).round(3))

# %%
import pandas as pd
import matplotlib.pyplot as plt

risk_percent = pd.crosstab(
    clean["Program"],
    clean["Risk_Label"],
    normalize="index"
) * 100

risk_percent = risk_percent.reindex(
    columns=["At Risk", "Not At Risk"]
)

ax = risk_percent.plot(
    kind="bar",
    stacked=True,
    figsize=(9, 5),
    color=["#E57373", "#64B5F6"]
)

plt.title("Academic Risk Distribution by Program")
plt.xlabel("Academic Program")
plt.ylabel("Percentage of Students (%)")
plt.xticks(rotation=0)
plt.ylim(0, 100)
plt.legend(title="Risk Label")

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f%%",
        label_type="center"
    )

plt.tight_layout()
plt.savefig("risk_distribution_by_program.png", dpi=300)
plt.show()


# %%


