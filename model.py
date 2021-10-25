import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('data/train.csv')

# as I am using this to test dashboarding I am not too worried about the data used or accuracy
df = df.drop(['Cabin'], axis=1)
df = df.dropna(axis=0)

df = df.rename(columns={
        'Pclass' : 'Passenger Class',
        'Parch' : 'Number of Parents/Children',
        'SibSp' : 'Number of Siblings',
        'Embark' : 'Port Embarked From'
    })

def make_model(features, neighbours):

    # classifier is random forest
    classifier = KNeighborsClassifier(n_neighbors=neighbours)

    # splitting training and test data 80% training 20% test
    X_train, X_test, y_train, y_test = train_test_split(df[features], df['Survived'], test_size=0.2, random_state=42)
    # functions to use in pipeline to select the right data type for each category
    categorical = ['Passenger Class', 'Number of Siblings', 'Number of Parents/Children', 'Port Embarked From', 'Sex']
    continuous = ['Age', 'Fare']

    categorical_features = []
    continuous_features = []

    for f in features:
        if f in categorical:
            categorical_features.append(f)
        elif f in continuous:
            continuous_features.append(f)

    get_cat = FunctionTransformer(lambda x: x[categorical_features], validate=False)
    get_num = FunctionTransformer(lambda x: x[continuous_features], validate=False)

    if len(categorical_features) == 0:
        #  full model pipeline to do pre-processing and prediction
        full_line = Pipeline([
            # feature union to combine all the data types
            ('union', FeatureUnion([
                ('numeric', Pipeline([
                    ('select', get_num),
                    ('impute', SimpleImputer())
                ]))
            ])),
            ('classify', classifier)
        ])

        full_line.fit(X_train, y_train)

        preds = full_line.predict(X_test)
        accuracy = accuracy_score(y_test, preds)

        matrix = confusion_matrix(y_test, preds)

        return accuracy, matrix

    elif len(continuous_features) == 0:
        #  full model pipeline to do pre-processing and prediction
        full_line = Pipeline([
            # feature union to combine all the data types
            ('union', FeatureUnion([
                ('cat', Pipeline([
                    ('select', get_cat),
                    ('encode', OneHotEncoder())  #  one hot encode categorical variables to make numerical
                ]))
            ])),
            ('classify', classifier)
        ])

        full_line.fit(X_train, y_train)

        preds = full_line.predict(X_test)
        accuracy = accuracy_score(y_test, preds)

        matrix = confusion_matrix(y_test, preds)

        return accuracy, matrix
    else:
        #  full model pipeline to do pre-processing and prediction
        full_line = Pipeline([
            # feature union to combine all the data types
            ('union', FeatureUnion([
                ('numeric', Pipeline([
                    ('select', get_num),
                    ('impute', SimpleImputer())
                ])),
                ('cat', Pipeline([
                    ('select', get_cat),
                    ('encode', OneHotEncoder())  #  one hot encode categorical variables to make numerical
                ]))
            ])),
            ('classify', classifier)
        ])

        full_line.fit(X_train, y_train)

        preds = full_line.predict(X_test)
        accuracy = accuracy_score(y_test, preds)

        matrix = confusion_matrix(y_test, preds)

        return accuracy, matrix
