import streamlit as st
import plotly.express as px
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model import df, make_model

"""
# An Unsinkable Dashboard

*My first Streamlit app will use titanic data to test out features and visualisations*

## First stop: EDA

Passengers embarking here can play around with some charts and change the variables to see what items they think have 
the greatest effect on survival rate 

First up the discrete data: Sex, Number of siblings, Number of parents and Embarked port. Filter which data item you 
would like to see and draw inferences as you may. 

"""
bar_df = df
bar_df['Passenger Class'] = bar_df['Passenger Class'].astype(str)
bar_df['Number of Parents/Children'] = bar_df['Number of Parents/Children'].astype(str)
bar_df['Number of Siblings'] = bar_df['Number of Siblings'].astype(str)
columns = ['Sex', 'Port Embarked From', 'Passenger Class', 'Number of Parents/Children', 'SibSp']
bar_selected_column = st.selectbox("select column", columns)
cols_to_keep = [bar_selected_column, 'Survived', 'PassengerId']

bar_data = bar_df[cols_to_keep]

bar_data = bar_data.groupby([bar_selected_column, 'Survived']).PassengerId.count().reset_index()

bar = px.bar(bar_data, x="Survived", color=bar_selected_column,
             y='PassengerId',
             title="Number of passengers by survival and " + bar_selected_column.lower(),
             barmode='group',
             template='seaborn',
             labels={
                     "PassengerId": "Number of passengers",
                     "Survived": "Survived?",
                 },
             )

st.write(bar)

"""
Next up let's look at our numeric data.

Here we will try to look out for 'survival clusters' plotting two different continuous variables and colouring the points
by survival. There are only actually two pieces of continuous data in the dataset so to fully test out streamlit's
functionality I have included some discrete data to give more options to the drop downs. Fare and Age are the only
truly continuous variables so are shown by default.

First select something to plot on the x-axis:
"""
columns_scatter_x = ['Age', 'Number of Siblings', 'Number of Parents/Children', 'Fare']
x_axis_scatter_column = st.selectbox("select column", columns_scatter_x)
"""
Next select the data we should plot on the y-axis:
"""
columns_scatter_y = ['Age', 'Fare', 'Number of Siblings', 'Number of Parents/Children']
columns_scatter_y.remove(x_axis_scatter_column)
y_axis_scatter_column = st.selectbox("select column", columns_scatter_y)

scatter = px.scatter(
    df,
    x=x_axis_scatter_column,
    y=y_axis_scatter_column,
    color='Survived',
    template='seaborn',
    title= 'Survival of passengers by ' + x_axis_scatter_column + " and " + y_axis_scatter_column
)

st.write(scatter)

"""
Hopefully this brief section on EDA has been useful but feel free to take a return trip through the charts whenever
you need to! Now model ahoy!

## Next port: Modelling

The modelling will be done to predict survival. The model will use a simple K Nearest Neighbours Classifier at this 
stage.

The default features selected are sex, age, fare and passenger class but feel free to select more/different features.

You can also edit the number of nearest neighbours used in the algorithm and see how this and the feature changes
affect model accuracy.
"""
features = ['Passenger Class', 'Number of Siblings', 'Number of Parents/Children', 'Port Embarked From', 'Sex', 'Age', 'Fare']

model_features = st.multiselect(label='Select modelling features:',
                                default=['Age', 'Fare', 'Passenger Class', 'Sex'],
                                options=features)

if len(model_features) < 1:
    st.error('Please enter at least one feature to model.')

neighbours = st.slider(label='Select the number of nearest neighbours:',
                       min_value=1,
                       max_value=12)

modelling = make_model(model_features, neighbours)

accuracy = modelling[0]
matrix = modelling[1]

"""
The accuracy of the current model is: 
"""
st.write(accuracy)

"""
You can view the confusion matrix below for a more in depth view of this classification:
"""


group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in matrix.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in matrix.flatten()/np.sum(matrix)]

labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2,2)

fig, ax = plt.subplots()
ax = sns.heatmap(matrix, annot=labels, fmt='')
st.pyplot(fig)

"""
## Final destination: Conclusion

Finding the best model accuracy is tough with so many features to choose from but it could be so much harder!

In future I hope to build on this project by allowing passengers to choose their own model at the end to see how high
we can boost that accuracy!

Until then it's time we parted ways but I'm sure we will embark again soon, this project has been really useful for me
to get some hands on Streamlit experience and hopefully this isn't the end of that journey.

- Joe Lewis
"""