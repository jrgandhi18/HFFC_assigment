from supabase import create_client, Client

SUPABASE_URL = "https://iptafzkllfxcvlsxcmeq.supabase.co"
SUPABASE_KEY = "sb_publishable_ZPxrye2bVpJWnLxrFG21yQ_G9Ee6fFQ"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def insert_lead(name: str, income: int):
    try:
        # Sirf wahi columns bhejo jo 100% table mein hain
        data = {
            "customer_name": name,
            "monthly_income": income
        }
        
        response = supabase.table("loan_leads").insert(data).execute()
        print("Bhai, Success! Data inserted:", response.data)
        
    except Exception as e:
        print("Abhi bhi error hai:", str(e))

if __name__ == "__main__":
    insert_lead("Jay Gandhi", 75000)