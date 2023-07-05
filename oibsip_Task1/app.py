import streamlit as st
import pickle


def main():
    st.title("Iris Classification with Logistic Regression")
    st.write("Enter the values to make a prediction on the Iris species.")

    # Load the trained model
    with open('pipe.pkl', 'rb') as f:
        model = pickle.load(f)

    # Take input from the user
    sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, step=0.1)
    sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, step=0.1)
    petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, step=0.1)
    petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, step=0.1)

    if st.button("Predict"):
        # Make a prediction
        features = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = model.predict(features)
        species_mapping = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
        predicted_species = species_mapping[prediction[0]]

        # Display the prediction
        st.subheader("Prediction")
        st.write(f"The predicted species is: {predicted_species}")


if __name__ == "__main__":
    main()

