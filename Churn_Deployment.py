import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import KNNImputer
from keras.models import load_model
import pickle

# Load the trained model
best_model = load_model("/Users/faithmwinnominusah/Downloads/best_model.h5")

# Load the feature names used during training
chosen_features = [
    "TotalCharges", "MonthlyCharges", "tenure", "Contract_Month-to-month",
    "OnlineSecurity_No", "gender", "PaymentMethod_Electronic check", "", "TechSupport_No", "PaperlessBilling","Partner"
]

# Create a Streamlit app 
st.title("Churn Prediction App")

# Collect user input for a new customer
#st.sidebar.header("Enter Customer Information")

def main():
    st.title("User Input Form")

    total_charges = st.number_input("Enter Total Charges", min_value=0.0, step=1.0)
    monthly_charges = st.number_input("Enter Monthly Charges", min_value=0.0, step=1.0)
    tenure = st.number_input("Enter tenure (in months)", min_value=0, step=1)

    # Input fields for strings
    contract_type = st.selectbox("Select Contract Type", ["Month-to-month", "Other"])
    online_security = st.selectbox("Select Online Security", ["No", "Other"])
    gender = st.radio("Select Gender", ["Male", "Female"])
    payment_method = st.selectbox("Select Payment Method", ["Electronic check", "Other"])
    tech_support = st.selectbox("Select Tech Support", ["No", "Other"])
    paperless_billing = st.checkbox("Paperless Billing")
    partner = st.checkbox("Partner")



    user_inputs = pd.DataFrame({
        "TotalCharges": [total_charges],
        "MonthlyCharges": [monthly_charges],
        "tenure": [tenure],
        "Contract_Month-to-month": [1 if contract_type == "Month-to-month" else 0],
        "OnlineSecurity_No": [1 if online_security == "No" else 0],
        "gender": [1 if gender == "Male" else 0],
        "PaymentMethod_Electronic check": [1 if payment_method == "Electronic check" else 0],
        "TechSupport_No": [1 if tech_support == "No" else 0],
        "PaperlessBilling": [1 if paperless_billing else 0],
        "Partner": [1 if partner else 0]
    })

    # Display the user inputs
    st.subheader("User Inputs:")
    st.write(f"Total Charges: ${total_charges}")
    st.write(f"Monthly Charges: ${monthly_charges}")
    st.write(f"Tenure: {tenure} months")
    st.write(f"Contract Type: {contract_type}")
    st.write(f"Online Security: {online_security}")
    st.write(f"Gender: {gender}")
    st.write(f"Payment Method: {payment_method}")
    st.write(f"Tech Support: {tech_support}")
    st.write(f"Paperless Billing: {paperless_billing}")
    st.write(f"Partner: {partner}")

   
    # Scale numerical features using StandardScaler
    with open("/Users/faithmwinnominusah/Downloads/scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    new_data_scaled = scaler.fit_transform( user_inputs)
    
    if st.button("Predict Churn"):
        # Make prediction
        prediction = best_model.predict(new_data_scaled)[0][0]
        if prediction <= 0.5:
            confidence_factor = 1 - prediction
        else:
            confidence_factor = round(prediction,2)
            print(confidence_factor)

   

    # Convert confidence factor to scalar before formatting
        confidence_factor_scalar = confidence_factor

        st.subheader("User Inputs:")
        st.write(f"Tenure: {tenure} months")
        st.write(f"Monthly Charges: ${monthly_charges}")
       

    # Display prediction result
        st.subheader("Prediction Result")
        if prediction <= 0.5:
            st.success("No Churn")
        else:
            st.error("Churn")

            
    # Display confidence factor
        st.subheader("Confidence Factor")
        st.write(f"The confidence factor of the model is: {confidence_factor_scalar:}")

if __name__ == "__main__":
    main()



