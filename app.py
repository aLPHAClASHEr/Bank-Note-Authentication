import numpy as np
import pickle
import streamlit as st
#import ast

loaded_model = pickle.load(open('banknote_model.pkl', 'rb'))
#classifier = pickle.load(loaded_model)


def welcome():
    return 'Welcome All'


def predict_note(input_data):
    new_array = np.asarray(input_data).reshape(1, -1)
    # There was an error that String cannot be be converted to float
    # I had used this line
    # new_array = np.array('input_data').reshape(1, -1)
    # The error was I placed quotes on input_data

    prediction = loaded_model.predict(new_array)

    if prediction[0] == 1:
        return'This is an original Note'
    else:
        return'The Note is a fake'

# variance,skewness,curtosis,entropy


def main():

    st.title('Banknote Authentication Website')

    variance = st.text_input('Enter Variance:')
    skewness = st.text_input('Enter Skewness:')
    curtosis = st.text_input('Enter Curtosis:')
    entropy = st.text_input('Enter Entropy:')

    predict = ''
    if st.button('Predict'):
        predict = predict_note([variance, skewness, curtosis, entropy])

    st.success(predict)


# There was an error where the screen was not loading the main function
# It was corrected by putting the bellow 2 lines outside the main function block
if __name__ == '__main__':
    main()
