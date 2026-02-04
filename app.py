import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, text
from datetime import datetime, date
import re
# ---------------- CONFIG ----------------

DB_USER = "postgres"
DB_PASSWORD = "1234"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "Validated"

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

st.set_page_config(page_title="Retail Raw Manager", layout="wide")
st.title("🛒 Retail Raw Data Manager")

# ---------------- LOAD AI MODEL ----------------
@st.cache_resource
def load_prediction_model():
    try:
        # Load the model and the encoder saved from the training step
        model = joblib.load("C:\\Users\\geyan\\OneDrive\\Desktop\\hackathon\\rf_expenditure_model.pth")
        encoder = joblib.load("C:\\Users\\geyan\\OneDrive\\Desktop\\hackathon\\loyalty_encoder.joblib")
        return model, encoder
    except FileNotFoundError:
        return None, None

model, encoder = load_prediction_model()

# ============================
# SECTION 1 — VIEW TABLES
# ============================

st.header("📂 View Database Tables")

try:
    tables = pd.read_sql("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema='public'
    ORDER BY table_name;
    """, engine)
    
    table_list = tables["table_name"].tolist()
    selected_table = st.selectbox("Select table to view", table_list)

    if selected_table:
        # Fetch the data
        df = pd.read_sql(f'SELECT * FROM "{selected_table}" LIMIT 1000', engine)
        
        # --- FIX: Convert object columns to string to ensure visibility ---
        # This forces the frontend to render them as simple text, which fixes contrast issues
        df = df.astype(str) 
        
        st.dataframe(df, use_container_width=True)
        st.caption(f"Rows shown: {len(df)}")

except Exception as e:
    st.error(f"Database connection error: {e}")

st.divider()
# ============================
# SECTION 2 — SMART INSERT SYSTEM (Master Data + Transaction)
# ============================

st.header("➕ Universal Data Entry")

# --- 1. Load Master Data ---
@st.cache_data
def get_master_data():
    try:
        # We use strict SQL to ensure we get fresh data every time cache clears
        s_df = pd.read_sql("SELECT * FROM stores", engine)
        p_df = pd.read_sql("SELECT * FROM products", engine)
        c_df = pd.read_sql("SELECT * FROM customer_details", engine)
        # NEW: Load Promotions
        promo_df = pd.read_sql("SELECT * FROM promotion_details", engine)
        return s_df, p_df, c_df, promo_df
    except Exception as e:
        # Returns empty DFs if tables don't exist yet
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

stores_df, products_df, customers_df, promo_df = get_master_data()

# --- 2. Input Section Layout ---
st.info("ℹ️ Select existing data to auto-fill, or select **'➕ New...'** to create entries on the fly.")

# We use tabs to keep the interface clean while handling complex data
tab_store, tab_prod, tab_cust, tab_promo, tab_txn = st.tabs([
    "🏢 Store", "📦 Product", "👤 Customer", "🏷️ Promotion", "📝 Transaction Finalize"
])

# ==========================
# TAB 1: STORE SELECTION
# ==========================
with tab_store:
    # Create a list that includes the "New" option
    store_opts = ["➕ New Store"] + stores_df['store_id'].tolist() if not stores_df.empty else ["➕ New Store"]
    
    sel_store_id = st.selectbox("Select Store", store_opts)
    
    # Initialize variables
    final_store_id = ""
    final_store_name = ""
    final_store_city = ""
    final_store_region = ""
    is_new_store = False

    if sel_store_id == "➕ New Store":
        is_new_store = True
        st.markdown("---")
        st.write("**Create New Store**")
        col1, col2 = st.columns(2)
        new_store_id = col1.text_input("New Store ID (e.g., S099)")
        new_store_name = col2.text_input("Store Name")
        col1, col2 = st.columns(2)
        new_store_city = col1.text_input("City")
        new_store_region = col2.text_input("Region")
        
        # Assign to final vars
        final_store_id = new_store_id
        final_store_name = new_store_name
        final_store_city = new_store_city
        final_store_region = new_store_region
    else:
        # Existing Store Logic
        row = stores_df[stores_df['store_id'] == sel_store_id].iloc[0]
        final_store_id = row['store_id']
        final_store_name = row['store_name']
        final_store_city = row['store_city']
        final_store_region = row['store_region']
        st.success(f"📍 Selected: **{final_store_name}** ({final_store_city})")

# ==========================
# TAB 2: PRODUCT SELECTION
# ==========================
with tab_prod:
    prod_opts = ["➕ New Product"] + products_df['product_id'].tolist() if not products_df.empty else ["➕ New Product"]
    sel_prod_id = st.selectbox("Select Product", prod_opts)
    
    is_new_prod = False
    
    if sel_prod_id == "➕ New Product":
        is_new_prod = True
        st.markdown("---")
        st.write("**Create New Product**")
        col1, col2 = st.columns(2)
        new_prod_id = col1.text_input("New Product ID")
        new_prod_name = col2.text_input("Product Name")
        col1, col2 = st.columns(2)
        new_prod_cat = col1.text_input("Category")
        new_prod_price = col2.number_input("Unit Price", min_value=0.0, step=0.5)
        
        final_prod_id = new_prod_id
        final_prod_name = new_prod_name
        final_prod_cat = new_prod_cat
        final_prod_price = new_prod_price
    else:
        row = products_df[products_df['product_id'] == sel_prod_id].iloc[0]
        final_prod_id = row['product_id']
        final_prod_name = row['product_name']
        final_prod_cat = row['product_category']
        final_prod_price = float(row['unit_price'])
        st.success(f"📦 Selected: **{final_prod_name}** (Rs : {final_prod_price})")

# ==========================
# TAB 3: CUSTOMER SELECTION
# ==========================
with tab_cust:
    cust_opts = ["➕ New Customer"] + customers_df['customer_id'].tolist() if not customers_df.empty else ["➕ New Customer"]
    sel_cust_id = st.selectbox("Select Customer", cust_opts)
    
    is_new_cust = False
    
    if sel_cust_id == "➕ New Customer":
        is_new_cust = True
        st.markdown("---")
        st.write("**Register New Customer**")
        col1, col2, col3 = st.columns(3)
        new_cust_id = col1.text_input("New Customer ID")
        new_cust_name = col2.text_input("First Name")
        new_cust_email = col3.text_input("Email Address")
        
        col1, col2, col3 = st.columns(3)
        new_cust_phone = col1.text_input("Phone (10 digits)")
        new_cust_loyalty = col2.selectbox("Loyalty Tier", ["Bronze", "Silver", "Gold", "Platinum"])
        new_cust_since = col3.date_input("Joined Date", value=date.today())
        
        final_cust_id = new_cust_id
        final_cust_name = new_cust_name
        final_cust_email = new_cust_email
        final_cust_phone = new_cust_phone
        final_cust_loyalty = new_cust_loyalty
        final_cust_since = new_cust_since
    else:
        row = customers_df[customers_df['customer_id'] == sel_cust_id].iloc[0]
        final_cust_id = row['customer_id']
        final_cust_name = row['first_name']
        final_cust_email = row['email']
        final_cust_phone = row['customer_phone']
        final_cust_loyalty = row['loyalty_status']
        final_cust_since = pd.to_datetime(row['customer_since']).date()
        st.success(f"👤 Selected: **{final_cust_name}** ({final_cust_loyalty})")

# ==========================
# TAB 4: PROMOTION SELECTION (NEW!)
# ==========================
with tab_promo:
    # Default option is "No Promotion"
    promo_opts = ["❌ No Promotion", "➕ New Promotion"] + promo_df['promotion_id'].tolist() if not promo_df.empty else ["❌ No Promotion", "➕ New Promotion"]
    sel_promo_id = st.selectbox("Select Promotion", promo_opts)
    
    is_new_promo = False
    final_promo_id = None
    final_promo_name = None
    final_discount_pct = 0.0
    
    if sel_promo_id == "➕ New Promotion":
        is_new_promo = True
        st.markdown("---")
        st.write("**Create New Promotion**")
        col1, col2 = st.columns(2)
        new_promo_id = col1.text_input("New Promo ID")
        new_promo_name = col2.text_input("Promo Name")
        
        col1, col2, col3 = st.columns(3)
        new_promo_start = col1.date_input("Start Date")
        new_promo_end = col2.date_input("End Date")
        new_promo_disc = col3.number_input("Discount %", min_value=0.0, max_value=100.0, step=1.0)
        
        final_promo_id = new_promo_id
        final_promo_name = new_promo_name
        final_discount_pct = new_promo_disc
        
    elif sel_promo_id == "❌ No Promotion":
        final_promo_id = None
        final_promo_name = None
        final_discount_pct = 0.0
        
    else:
        row = promo_df[promo_df['promotion_id'] == sel_promo_id].iloc[0]
        final_promo_id = row['promotion_id']
        final_promo_name = row['promotion_name']
        final_discount_pct = float(row['discount_percentage'])
        
        # Validation visual aid
        p_start = pd.to_datetime(row['start_date']).date()
        p_end = pd.to_datetime(row['end_date']).date()
        st.success(f"🏷️ **{final_promo_name}**: {final_discount_pct}% Off (Valid: {p_start} to {p_end})")

# ==========================
# TAB 5: FINALIZE TRANSACTION
# ==========================
with tab_txn:
    st.subheader("📝 Transaction Details")
    
    with st.form("final_insert_form"):
        col1, col2, col3 = st.columns(3)
        txn_id = col1.text_input("Transaction ID")
        txn_date = col2.date_input("Transaction Date", value=date.today())
        quantity = col3.number_input("Quantity", min_value=1, step=1)
        
        st.markdown(f"**Summary:** Selling **{quantity}** x **{final_prod_name}** to **{final_cust_name}** at **{final_store_name}**.")
        if final_promo_id:
            st.markdown(f"🔥 Applying Promo: **{final_promo_name}** ({final_discount_pct}%)")
        
        submitted = st.form_submit_button("✅ Submit All Data")

        if submitted:
            import re
            errors = []
            
            # --- 1. Basic Validations ---
            if not txn_id: errors.append("Transaction ID is required.")
            if is_new_store and not final_store_id: errors.append("New Store ID missing.")
            if is_new_cust and not final_cust_id: errors.append("New Customer ID missing.")
            if is_new_prod and not final_prod_id: errors.append("New Product ID missing.")
            
            # --- 2. Promo Validation (If selected) ---
            if final_promo_id and not is_new_promo:
                 # Check if transaction date is within promo range
                 if not (p_start <= txn_date <= p_end):
                     st.warning(f"⚠️ Warning: Transaction date {txn_date} is outside promotion range ({p_start} to {p_end}). Applying anyway.")

            if errors:
                for e in errors: st.error(f"❌ {e}")
            else:
                try:
                    with engine.begin() as conn:
                        # --- 3. CASCADING INSERTS (Parents First) ---
                        
                        # A. Insert New Store?
                        if is_new_store:
                            conn.execute(text("INSERT INTO stores VALUES (:id, :name, :city, :region)"),
                                         {"id": final_store_id, "name": final_store_name, "city": final_store_city, "region": final_store_region})
                            st.toast(f"✅ Created Store {final_store_id}")

                        # B. Insert New Product?
                        if is_new_prod:
                            conn.execute(text("INSERT INTO products VALUES (:id, :name, :cat, :price)"),
                                         {"id": final_prod_id, "name": final_prod_name, "cat": final_prod_cat, "price": final_prod_price})
                            st.toast(f"✅ Created Product {final_prod_id}")
                            
                        # C. Insert New Customer?
                        if is_new_cust:
                            # Note: last_purchase_date set to None or today for new user
                            conn.execute(text("""
                                INSERT INTO customer_details (customer_id, first_name, email, customer_phone, loyalty_status, customer_since)
                                VALUES (:id, :name, :email, :phone, :loyalty, :since)
                            """), {
                                "id": final_cust_id, "name": final_cust_name, "email": final_cust_email,
                                "phone": final_cust_phone, "loyalty": final_cust_loyalty, "since": final_cust_since
                            })
                            st.toast(f"✅ Registered Customer {final_cust_id}")

                        # D. Insert New Promotion?
                        if is_new_promo:
                            conn.execute(text("""
                                INSERT INTO promotion_details (promotion_id, promotion_name, start_date, end_date, discount_percentage)
                                VALUES (:id, :name, :start, :end, :disc)
                            """), {
                                "id": final_promo_id, "name": final_promo_name, 
                                "start": new_promo_start, "end": new_promo_end, "disc": final_discount_pct
                            })
                            st.toast(f"✅ Created Promo {final_promo_id}")

                        # --- 4. CALCULATIONS ---
                        total_price = final_prod_price * quantity
                        discount_amt = total_price * (final_discount_pct / 100)
                        final_amount = total_price - discount_amt
                        
                        month = txn_date.month
                        day_of_week = txn_date.isoweekday()
                        tenure = (txn_date - final_cust_since).days

                        # --- 5. FINAL TRANSACTION INSERT ---
                        insert_query = text("""
                            INSERT INTO retail_raw VALUES (
                                :transaction_id, :transaction_date, :store_id, :store_name, :store_city, :store_region,
                                :customer_id, :first_name, :email, :customer_phone, :loyalty_status, :customer_since,
                                :product_id, :product_name, :product_category, :unit_price, :quantity,
                                :promotion_id, :promotion_name, :discount_percentage,
                                :total_price, :discount_amount, :final_amount,
                                :month, :day_of_week, :customer_tenure_days
                            )
                        """)
                        
                        params = {
                            "transaction_id": txn_id, "transaction_date": txn_date,
                            "store_id": final_store_id, "store_name": final_store_name, "store_city": final_store_city, "store_region": final_store_region,
                            "customer_id": final_cust_id, "first_name": final_cust_name, "email": final_cust_email, "customer_phone": final_cust_phone,
                            "loyalty_status": final_cust_loyalty, "customer_since": final_cust_since,
                            "product_id": final_prod_id, "product_name": final_prod_name, "product_category": final_prod_cat, "unit_price": final_prod_price,
                            "quantity": quantity,
                            "promotion_id": final_promo_id, "promotion_name": final_promo_name, "discount_percentage": final_discount_pct,
                            "total_price": total_price, "discount_amount": discount_amt, "final_amount": final_amount,
                            "month": month, "day_of_week": day_of_week, "customer_tenure_days": tenure
                        }
                        
                        conn.execute(insert_query, params)
                    
                    st.success(f"🎉 SUCCESS! Transaction {txn_id} recorded completely.")
                    st.balloons()
                    # Refresh Cache so new items appear in dropdowns next time
                    st.cache_data.clear()

                except Exception as e:
                    st.error(f"❌ Database Error: {e}")
st.divider()               

# ============================
# SECTION 3 — DELETE retail_raw
# ============================

st.header("🗑 Delete Row from retail_raw")

delete_txn = st.text_input("Enter transaction_id to delete")

if st.button("Delete Transaction"):
    delete_query = text("DELETE FROM retail_raw WHERE transaction_id = :t")

    with engine.begin() as conn:
        conn.execute(delete_query, {"t": delete_txn})

    st.success(f"🗑 Deleted transaction {delete_txn}")

st.divider()

# ============================
# SECTION 4 — AI PREDICTION
# ============================

st.header("🔮 AI Customer Spend Prediction")

if model is None:
    st.warning("⚠️ Model files (`rf_expenditure_model.pth`, `loyalty_encoder.joblib`) not found. Please upload them to run predictions.")
else:
    # 1. Select Customer
    try:
        # Get list of customers from DB
        cust_list_query = "SELECT DISTINCT customer_id FROM retail_raw ORDER BY customer_id"
        all_customers = pd.read_sql(cust_list_query, engine)['customer_id'].tolist()
        
        selected_customer = st.selectbox("Select Customer to Predict", all_customers)
        
        if st.button("Predict Next Month Spend"):
            
            # 2. Get Data for this Customer
            data_query = text("SELECT * FROM retail_raw WHERE customer_id = :cid")
            
            with engine.connect() as conn:
                cust_df = pd.read_sql(data_query, conn, params={"cid": selected_customer})

            if not cust_df.empty:
                # 3. Preprocess Features
                cust_df['transaction_date'] = pd.to_datetime(cust_df['transaction_date'])
                
                # --- UPDATED: Use Real-Time for Snapshot ---
                snapshot_date = pd.to_datetime("today")
                
                # --- Feature Engineering ---
                # A. Recency (Days since last purchase relative to TODAY)
                recency = (snapshot_date - cust_df['transaction_date'].max()).days
                
                # B. Frequency
                frequency = len(cust_df)
                
                # C. Monetary
                monetary = cust_df['final_amount'].sum()
                
                # D. Tenure
                tenure = cust_df['customer_tenure_days'].mean()
                if tenure == 0: tenure = 1 
                
                # E. Purchase Velocity
                purchase_velocity = frequency / (tenure / 30)
                
                # F. Loyalty Encoded
                loyalty_status = cust_df['loyalty_status'].iloc[0]
                try:
                    loyalty_encoded = encoder.transform([loyalty_status])[0]
                except ValueError:
                    loyalty_encoded = 0 
                
                # 4. Create Feature Vector
                features = pd.DataFrame([[
                    recency, frequency, monetary, tenure, purchase_velocity, loyalty_encoded
                ]], columns=['Recency', 'Frequency', 'Monetary', 'customer_tenure_days', 'purchase_velocity', 'loyalty_encoded'])
                
                # 5. Predict
                prediction = model.predict(features)[0]
                
                # 6. Display
                st.subheader(f"💰 Predicted Spend: Rs:{prediction:.2f}")
                
                with st.expander("See Feature Details"):
                    st.write("Current Metrics:")
                    st.dataframe(features)
                    st.info(f"Recency is calculated as days from Today ({snapshot_date.date()}) to last purchase.")

            else:
                st.error("No transaction history found for this customer.")

    except Exception as e:
        st.error(f"Error fetching customer data: {e}")

st.divider()

# ============================
# LIVE RAW PREVIEW
# ============================

st.header("📄 Latest retail_raw rows")

preview = pd.read_sql("""
SELECT * FROM retail_raw 
ORDER BY transaction_date DESC 
LIMIT 20
""", engine)

st.dataframe(preview, use_container_width=True)