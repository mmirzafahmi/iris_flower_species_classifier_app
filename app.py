import streamlit as st

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42
)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

st.title("Iris Flower Species Classification App")
st.write("This app classifies iris flower species based on sepal and petal measurements.")

sepal_length = st.slider("Sepal Length", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
sepal_width = st.slider("Sepal Width", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
petal_length = st.slider("Petal Length", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
petal_width = st.slider("Petal Width", min_value=0.0, max_value=10.0, value=0.5, step=0.1)

if st.button("Classify"):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = knn.predict(features)
    species = iris.target_names[prediction][0]
    st.write(f"The predicted species is: {species}")
