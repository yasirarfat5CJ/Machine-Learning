import streamlit as st

from src.predict import predict_message


st.title("Email / SMS Spam Classifier")

input_sms = st.text_area(
    "Enter a message to check:"
)


if st.button("Predict"):

    if not input_sms.strip():

        st.warning("Please enter a message first.")

    else:

        result = predict_message(input_sms)

        # Display prediction
        if result["prediction"] == 1:
            st.error("SPAM Message Detected!")
        else:
            st.success("This message is NOT SPAM.")

        # Debug information
        st.write("### Debug Information")

        st.write(
            "Preprocessed text:",
            result["transformed_text"]
        )

        st.write(
            "Number of TF-IDF features:",
            result["non_zero_features"]
        )

        st.write(
            "Recognized words:",
            result["recognized_words"]
        )

        st.write(
            "HAM probability:",
            result["ham_probability"]
        )

        st.write(
            "SPAM probability:",
            result["spam_probability"]
        )