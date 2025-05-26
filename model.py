import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

athlete = pd.read_excel("athlete.xlsx")
iq = pd.read_excel("IQ.xlsx")
performances = pd.read_excel("performances.xlsx")

df = iq.merge(athlete[['ID', 'Sex']], on='ID')
df = df.merge(performances[['ID', 'Season', 'Sport']], on='ID')

df['Medal_Won'] = df['Medal'].notna().astype(int)

le_sport = LabelEncoder()
le_academy = LabelEncoder()
le_sex = LabelEncoder()
le_season = LabelEncoder()

df['Sport'] = le_sport.fit_transform(df['Sport'])
df['Academy'] = le_academy.fit_transform(df['sports_academy'])
df['Sex'] = le_sex.fit_transform(df['Sex'])
df['Season'] = le_season.fit_transform(df['Season'])

features = [
    'Age', 'BMI', 'BMI_Score', 'Age_Score', 'AcademyIQ',
    'Raw_IQ', 'IQ_Final', 'Sex', 'Season', 'Sport', 'Academy'
]

X = df[features]
y = df['Medal_Won']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'olympic_model.pkl')
joblib.dump(le_sport, 'sport_encoder.pkl')
joblib.dump(le_academy, 'academy_encoder.pkl')
joblib.dump(le_sex, 'sex_encoder.pkl')
joblib.dump(le_season, 'season_encoder.pkl')