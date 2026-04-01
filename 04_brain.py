import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from supabase import create_client
from tools import check_eligibility, calculate_emi

# 1. Load Environment Variables
load_dotenv("api_key.env")

# 2. Setup Gemini
# Use the key you provided
genai.configure(api_key="AIzaSyAfze52Ple2Hy0mV3qOfg_qMnbOMdKpV_Y")

# 3. Setup Supabase
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

# 4. System Prompt from your original logic [cite: 8, 11, 19]
SYSTEM_PROMPT = """
You are a HomeFirst Loan Counselor. 
1. Language Lock: Detect user language (English, Hindi, Marathi, Tamil) and STICK to it. [cite: 11]
2. Collect: customer_name, monthly_income, loan_amount, and property_value. [cite: 9]
3. Once you have these, use the 'check_eligibility_tool'. [cite: 14, 15]
4. Do not calculate math yourself. 
5. If eligible and high intent, trigger: [HANDOFF TRIGGERED]. [cite: 37]
6. Out-of-Domain: Only talk about Home Loans. [cite: 19]
"""

# 5. Define the Tool Function for Gemini
def check_eligibility_tool(monthly_income: float, loan_amount: float, property_value: float):
    """Calculates home loan eligibility and EMI using business rules."""
    # Calling your exact tools.py logic [cite: 16, 18, 24]
    is_eligible, result_msg = check_eligibility(monthly_income, loan_amount, property_value)
    
    # Save to Supabase if eligible [cite: 37]
    if is_eligible:
        supabase.table("loan_leads").insert({
            "customer_name": "User",
            "monthly_income": monthly_income,
            "loan_amount": loan_amount,
            "status": "Eligible"
        }).execute()
        return f"{result_msg} [HANDOFF TRIGGERED]"
    
    return result_msg

# 6. Initialize Model with Function Support
# Model initialization update karo
model = genai.GenerativeModel(
    model_name='models/gemini-1.5-flash', # Pura path likhna safe hota hai
    tools=[check_eligibility_tool]
)

# Start chat with automatic function calling enabled
chat = model.start_chat(enable_automatic_function_calling=True)

def chat_with_ai(user_input):
    # Combining prompt and input
    full_query = f"{SYSTEM_PROMPT}\n\nUser: {user_input}"
    response = chat.send_message(full_query)
    return response.text

# --- TEST IT ---
if __name__ == "__main__":
    test_msg = "Mera naam Jay hai. Meri income 1 lakh hai, 20 lakh ka loan chahiye aur property 40 lakh ki hai."
    print("AI Response:", chat_with_ai(test_msg))
    