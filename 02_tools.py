import math

def calculate_emi(principal: float, annual_rate: float = 9.2, years: int = 10) -> int:
    """
    Standard EMI Formula: [P x R x (1+R)^N]/[(1+R)^N-1]
    Using values from the HomeFirst screenshot as defaults.
    """
    # Monthly interest rate
    r = (annual_rate / 12) / 100 
    # Total number of months
    n = years * 12 
    
    # EMI Calculation
    emi = (principal * r * (1 + r)**n) / ((1 + r)**n - 1)
    return round(emi)

def check_eligibility(monthly_income: float, loan_amount: float, property_value: float):
    """
    HFFC Business Rules based on Indian Home Loan standards.
    """
    # 1. LTV (Loan to Value) Rule: Usually max 80% [cite: 22, 23]
    ltv_ratio = (loan_amount / property_value) * 100
    if ltv_ratio > 80:
        return False, f"Rejected: Loan amount is {ltv_ratio:.1f}% of property value (Limit: 80%)."
    
    # 2. FOIR (Fixed Obligation to Income Ratio) Rule: Max 50% [cite: 22, 23]
    # Calculating EMI for the requested loan amount
    emi = calculate_emi(loan_amount)
    foir = (emi / monthly_income) * 100
    
    if foir > 50:
        return False, f"Rejected: Monthly EMI (₹{emi}) exceeds 50% of your income."
    
    return True, f"Eligible! Your estimated EMI will be ₹{emi}."

# --- Testing the logic with HomeFirst Screenshot values ---
if __name__ == "__main__":
    # Test 1: EMI Calculation (Should be around 6388)
    test_emi = calculate_emi(500000, 9.2, 10)
    print(f"Test 1 - EMI for 5L at 9.2% for 10yrs: {test_emi}")

    # Test 2: Eligibility Check
    # Scenario: 50k income, 5L loan, 10L property
    is_eligible, message = check_eligibility(50000, 500000, 1000000)
    print(f"Test 2 - Eligibility Result: {message}")