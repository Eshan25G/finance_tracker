import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime, timedelta
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="AI-Powered Personal Finance Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'expenses_df' not in st.session_state:
    st.session_state.expenses_df = pd.DataFrame()
if 'budgets' not in st.session_state:
    st.session_state.budgets = {}
if 'ai_insights' not in st.session_state:
    st.session_state.ai_insights = []

# Helper Functions
def categorize_expense_ai(description, amount):
    """AI-powered expense categorization"""
    description_lower = description.lower()
    
    # Advanced categorization rules
    food_keywords = ['restaurant', 'food', 'grocery', 'cafe', 'dining', 'pizza', 'uber eats', 'delivery', 'starbucks', 'mcdonald', 'subway']
    transport_keywords = ['gas', 'fuel', 'uber', 'taxi', 'bus', 'train', 'parking', 'toll', 'car', 'metro']
    entertainment_keywords = ['movie', 'netflix', 'spotify', 'game', 'theater', 'concert', 'bar', 'club', 'entertainment']
    shopping_keywords = ['amazon', 'store', 'mall', 'shopping', 'clothes', 'shoes', 'electronics', 'target', 'walmart']
    utilities_keywords = ['electricity', 'water', 'gas bill', 'internet', 'phone', 'mobile', 'utility']
    health_keywords = ['doctor', 'hospital', 'pharmacy', 'medicine', 'health', 'dental', 'medical']
    
    if any(keyword in description_lower for keyword in food_keywords):
        return 'Food & Dining'
    elif any(keyword in description_lower for keyword in transport_keywords):
        return 'Transportation'
    elif any(keyword in description_lower for keyword in entertainment_keywords):
        return 'Entertainment'
    elif any(keyword in description_lower for keyword in shopping_keywords):
        return 'Shopping'
    elif any(keyword in description_lower for keyword in utilities_keywords):
        return 'Utilities'
    elif any(keyword in description_lower for keyword in health_keywords):
        return 'Healthcare'
    elif amount > 500:
        return 'Major Purchase'
    else:
        return 'Other'

def generate_ai_insights(df):
    """Generate AI-powered financial insights"""
    insights = []
    
    if df.empty:
        return insights
    
    # Spending pattern analysis
    monthly_spending = df.groupby(df['Date'].dt.to_period('M'))['Amount'].sum()
    if len(monthly_spending) > 1:
        trend = "increasing" if monthly_spending.iloc[-1] > monthly_spending.iloc[-2] else "decreasing"
        insights.append(f"📈 Your monthly spending is {trend} compared to last month.")
    
    # Category analysis
    category_spending = df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
    top_category = category_spending.index[0]
    top_amount = category_spending.iloc[0]
    total_spending = df['Amount'].sum()
    percentage = (top_amount / total_spending) * 100
    
    insights.append(f"🏷️ Your highest spending category is '{top_category}' ({percentage:.1f}% of total spending).")
    
    # Unusual spending detection
    daily_spending = df.groupby(df['Date'].dt.date)['Amount'].sum()
    avg_daily = daily_spending.mean()
    high_spending_days = daily_spending[daily_spending > avg_daily * 2]
    
    if len(high_spending_days) > 0:
        insights.append(f"⚠️ You had {len(high_spending_days)} days with unusually high spending.")
    
    # Weekend vs weekday analysis
    df['Weekday'] = df['Date'].dt.day_name()
    weekend_spending = df[df['Weekday'].isin(['Saturday', 'Sunday'])]['Amount'].sum()
    weekday_spending = df[~df['Weekday'].isin(['Saturday', 'Sunday'])]['Amount'].sum()
    
    if weekend_spending > weekday_spending:
        insights.append("🎉 You spend more on weekends than weekdays. Consider planning weekend activities within budget.")
    
    return insights

def create_spending_clusters(df):
    """Create spending behavior clusters using ML"""
    if df.empty or len(df) < 5:
        return df
    
    # Prepare features for clustering
    features = df[['Amount']].copy()
    features['DayOfWeek'] = df['Date'].dt.dayofweek
    features['Month'] = df['Date'].dt.month
    features['Hour'] = df['Date'].dt.hour if 'Time' in df.columns else 12
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform clustering
    n_clusters = min(3, len(df))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Spending_Cluster'] = kmeans.fit_predict(features_scaled)
    
    # Label clusters
    cluster_labels = {0: 'Low Spender', 1: 'Medium Spender', 2: 'High Spender'}
    df['Cluster_Label'] = df['Spending_Cluster'].map(cluster_labels)
    
    return df

def create_pdf_report(df, budgets):
    """Create PDF report (placeholder - would need reportlab in real implementation)"""
    report_data = {
        'total_expenses': df['Amount'].sum(),
        'categories': df.groupby('Category')['Amount'].sum().to_dict(),
        'monthly_summary': df.groupby(df['Date'].dt.to_period('M'))['Amount'].sum().to_dict(),
        'budgets': budgets
    }
    return report_data

# Main App
def main():
    st.markdown('<h1 class="main-header">💰 AI-Powered Personal Finance Tracker</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    tab = st.sidebar.radio("Choose Section", 
                          ["📈 Dashboard", "📤 Upload Data", "💰 Budget Manager", "🤖 AI Insights", "📊 Analytics"])
    
    if tab == "📤 Upload Data":
        upload_data_section()
    elif tab == "💰 Budget Manager":
        budget_manager_section()
    elif tab == "🤖 AI Insights":
        ai_insights_section()
    elif tab == "📊 Analytics":
        analytics_section()
    else:
        dashboard_section()

def upload_data_section():
    st.header("📤 Upload Your Expense Data")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            # Display first few rows
            st.subheader("Preview of your data:")
            st.dataframe(df.head())
            
            # Data processing
            st.subheader("Data Processing")
            
            # Column mapping
            col1, col2, col3 = st.columns(3)
            with col1:
                date_col = st.selectbox("Date Column", df.columns)
            with col2:
                amount_col = st.selectbox("Amount Column", df.columns)
            with col3:
                desc_col = st.selectbox("Description Column", df.columns)
            
            if st.button("Process Data"):
                # Process the data
                processed_df = df.copy()
                processed_df['Date'] = pd.to_datetime(processed_df[date_col])
                processed_df['Amount'] = pd.to_numeric(processed_df[amount_col], errors='coerce')
                processed_df['Description'] = processed_df[desc_col].astype(str)
                
                # AI categorization
                with st.spinner("🤖 AI is categorizing your expenses..."):
                    processed_df['Category'] = processed_df.apply(
                        lambda x: categorize_expense_ai(x['Description'], x['Amount']), axis=1
                    )
                
                # Create spending clusters
                processed_df = create_spending_clusters(processed_df)
                
                # Store in session state
                st.session_state.expenses_df = processed_df[['Date', 'Amount', 'Description', 'Category', 'Cluster_Label']].copy()
                
                st.success("✅ Data processed successfully!")
                st.balloons()
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Manual entry
    st.subheader("➕ Add Manual Entry")
    with st.form("manual_entry"):
        col1, col2 = st.columns(2)
        with col1:
            date = st.date_input("Date", datetime.now())
            amount = st.number_input("Amount", min_value=0.01, step=0.01)
        with col2:
            description = st.text_input("Description")
            category = st.selectbox("Category", 
                                   ['Food & Dining', 'Transportation', 'Entertainment', 'Shopping', 
                                    'Utilities', 'Healthcare', 'Major Purchase', 'Other'])
        
        if st.form_submit_button("Add Entry"):
            new_entry = pd.DataFrame({
                'Date': [pd.to_datetime(date)],
                'Amount': [amount],
                'Description': [description],
                'Category': [category],
                'Cluster_Label': ['Manual Entry']
            })
            
            if not st.session_state.expenses_df.empty:
                st.session_state.expenses_df = pd.concat([st.session_state.expenses_df, new_entry], ignore_index=True)
            else:
                st.session_state.expenses_df = new_entry
            
            st.success("Entry added successfully!")

def budget_manager_section():
    st.header("💰 Budget Manager")
    
    # Set budgets
    st.subheader("Set Category Budgets")
    categories = ['Food & Dining', 'Transportation', 'Entertainment', 'Shopping', 
                  'Utilities', 'Healthcare', 'Major Purchase', 'Other']
    
    col1, col2 = st.columns(2)
    for i, category in enumerate(categories):
        with col1 if i % 2 == 0 else col2:
            budget_amount = st.number_input(f"{category} Budget", 
                                          min_value=0.0, 
                                          value=float(st.session_state.budgets.get(category, 0)),
                                          key=f"budget_{category}")
            st.session_state.budgets[category] = budget_amount
    
    # Budget vs Actual comparison
    if not st.session_state.expenses_df.empty:
        st.subheader("📊 Budget vs Actual Spending")
        
        # Calculate actual spending by category
        actual_spending = st.session_state.expenses_df.groupby('Category')['Amount'].sum()
        
        # Create comparison chart
        budget_comparison = []
        for category in categories:
            if st.session_state.budgets.get(category, 0) > 0:
                budget_comparison.append({
                    'Category': category,
                    'Budget': st.session_state.budgets[category],
                    'Actual': actual_spending.get(category, 0),
                    'Difference': st.session_state.budgets[category] - actual_spending.get(category, 0)
                })
        
        if budget_comparison:
            comparison_df = pd.DataFrame(budget_comparison)
            
            # Visualization
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Budget',
                x=comparison_df['Category'],
                y=comparison_df['Budget'],
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Bar(
                name='Actual',
                x=comparison_df['Category'],
                y=comparison_df['Actual'],
                marker_color='coral'
            ))
            
            fig.update_layout(
                title='Budget vs Actual Spending',
                xaxis_title='Category',
                yaxis_title='Amount ($)',
                barmode='group',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Budget alerts
            st.subheader("🚨 Budget Alerts")
            for _, row in comparison_df.iterrows():
                if row['Actual'] > row['Budget']:
                    st.markdown(f"""
                    <div class="warning-card">
                        <h4>⚠️ {row['Category']} Over Budget!</h4>
                        <p>Budget: ${row['Budget']:.2f} | Actual: ${row['Actual']:.2f}</p>
                        <p>Over by: ${row['Actual'] - row['Budget']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif row['Actual'] <= row['Budget'] * 0.8:
                    st.markdown(f"""
                    <div class="success-card">
                        <h4>✅ {row['Category']} Under Budget!</h4>
                        <p>Budget: ${row['Budget']:.2f} | Actual: ${row['Actual']:.2f}</p>
                        <p>Remaining: ${row['Budget'] - row['Actual']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)

def ai_insights_section():
    st.header("🤖 AI-Powered Financial Insights")
    
    if st.session_state.expenses_df.empty:
        st.warning("Please upload expense data first to generate insights.")
        return
    
    # Generate insights
    if st.button("🔄 Generate New Insights"):
        with st.spinner("🤖 AI is analyzing your spending patterns..."):
            st.session_state.ai_insights = generate_ai_insights(st.session_state.expenses_df)
    
    # Display insights
    if st.session_state.ai_insights:
        st.subheader("💡 Your Personalized Insights")
        for insight in st.session_state.ai_insights:
            st.markdown(f"""
            <div class="metric-card">
                <p>{insight}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Spending predictions
    st.subheader("🔮 Spending Predictions")
    if len(st.session_state.expenses_df) > 30:
        # Simple trend analysis
        monthly_spending = st.session_state.expenses_df.groupby(
            st.session_state.expenses_df['Date'].dt.to_period('M')
        )['Amount'].sum()
        
        if len(monthly_spending) >= 3:
            # Calculate trend
            trend = np.polyfit(range(len(monthly_spending)), monthly_spending.values, 1)[0]
            next_month_prediction = monthly_spending.iloc[-1] + trend
            
            st.metric(
                "Predicted Next Month Spending",
                f"${next_month_prediction:.2f}",
                f"${trend:.2f} trend"
            )
    
    # Anomaly detection
    st.subheader("🔍 Unusual Spending Detection")
    if len(st.session_state.expenses_df) > 10:
        df = st.session_state.expenses_df.copy()
        
        # Calculate Z-scores for amounts
        df['Amount_ZScore'] = np.abs((df['Amount'] - df['Amount'].mean()) / df['Amount'].std())
        anomalies = df[df['Amount_ZScore'] > 2]
        
        if not anomalies.empty:
            st.write("🚨 Unusual transactions detected:")
            st.dataframe(anomalies[['Date', 'Amount', 'Description', 'Category']].head())
        else:
            st.success("✅ No unusual spending patterns detected.")

def analytics_section():
    st.header("📊 Advanced Analytics")
    
    if st.session_state.expenses_df.empty:
        st.warning("Please upload expense data first to view analytics.")
        return
    
    df = st.session_state.expenses_df.copy()
    
    # Time series analysis
    st.subheader("📈 Spending Trends Over Time")
    
    # Daily spending
    daily_spending = df.groupby(df['Date'].dt.date)['Amount'].sum().reset_index()
    daily_spending.columns = ['Date', 'Amount']
    
    fig_daily = px.line(daily_spending, x='Date', y='Amount', 
                       title='Daily Spending Trend',
                       labels={'Amount': 'Amount ($)', 'Date': 'Date'})
    fig_daily.update_layout(height=400)
    st.plotly_chart(fig_daily, use_container_width=True)
    
    # Monthly spending
    monthly_spending = df.groupby(df['Date'].dt.to_period('M'))['Amount'].sum().reset_index()
    monthly_spending['Date'] = monthly_spending['Date'].astype(str)
    
    fig_monthly = px.bar(monthly_spending, x='Date', y='Amount',
                        title='Monthly Spending',
                        labels={'Amount': 'Amount ($)', 'Date': 'Month'})
    fig_monthly.update_layout(height=400)
    st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Category analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏷️ Spending by Category")
        category_spending = df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
        
        fig_pie = px.pie(values=category_spending.values, 
                        names=category_spending.index,
                        title='Spending Distribution by Category')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("📊 Category Spending Bars")
        fig_bar = px.bar(x=category_spending.index, 
                        y=category_spending.values,
                        title='Total Spending by Category',
                        labels={'y': 'Amount ($)', 'x': 'Category'})
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Heatmap analysis
    st.subheader("🔥 Spending Heatmap")
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['Month'] = df['Date'].dt.month_name()
    
    heatmap_data = df.groupby(['DayOfWeek', 'Month'])['Amount'].sum().unstack(fill_value=0)
    
    fig_heatmap = px.imshow(heatmap_data.values,
                           labels=dict(x="Month", y="Day of Week", color="Amount"),
                           x=heatmap_data.columns,
                           y=heatmap_data.index,
                           title="Spending Heatmap (Day vs Month)")
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Clustering analysis
    if 'Cluster_Label' in df.columns:
        st.subheader("🎯 Spending Behavior Clusters")
        
        cluster_summary = df.groupby('Cluster_Label').agg({
            'Amount': ['mean', 'sum', 'count']
        }).round(2)
        
        st.dataframe(cluster_summary)
        
        # Cluster visualization
        fig_cluster = px.scatter(df, x='Date', y='Amount', 
                               color='Cluster_Label',
                               title='Spending Patterns by Cluster',
                               labels={'Amount': 'Amount ($)'})
        st.plotly_chart(fig_cluster, use_container_width=True)

def dashboard_section():
    st.header("📈 Financial Dashboard")
    
    if st.session_state.expenses_df.empty:
        st.info("👋 Welcome! Please upload your expense data to get started.")
        st.markdown("""
        ### Getting Started:
        1. 📤 Go to **Upload Data** to upload your CSV file or add manual entries
        2. 💰 Set your budgets in **Budget Manager**
        3. 🤖 Get AI insights about your spending patterns
        4. 📊 Explore detailed analytics
        
        ### Sample CSV Format:
        ```
        Date,Amount,Description
        2024-01-15,25.50,Coffee Shop
        2024-01-16,150.00,Grocery Store
        2024-01-17,45.00,Gas Station
        ```
        """)
        return
    
    df = st.session_state.expenses_df.copy()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_expenses = df['Amount'].sum()
        st.metric("Total Expenses", f"${total_expenses:,.2f}")
    
    with col2:
        avg_daily = df.groupby(df['Date'].dt.date)['Amount'].sum().mean()
        st.metric("Avg Daily Spending", f"${avg_daily:.2f}")
    
    with col3:
        num_transactions = len(df)
        st.metric("Total Transactions", num_transactions)
    
    with col4:
        avg_transaction = df['Amount'].mean()
        st.metric("Avg Transaction", f"${avg_transaction:.2f}")
    
    # Quick visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Recent spending trend
        recent_spending = df.tail(30).groupby(df['Date'].dt.date)['Amount'].sum()
        fig_recent = px.line(x=recent_spending.index, y=recent_spending.values,
                           title='Recent Spending Trend (Last 30 transactions)',
                           labels={'y': 'Amount ($)', 'x': 'Date'})
        st.plotly_chart(fig_recent, use_container_width=True)
    
    with col2:
        # Top categories
        top_categories = df.groupby('Category')['Amount'].sum().nlargest(5)
        fig_top = px.bar(x=top_categories.values, y=top_categories.index,
                        orientation='h',
                        title='Top 5 Spending Categories',
                        labels={'x': 'Amount ($)', 'y': 'Category'})
        st.plotly_chart(fig_top, use_container_width=True)
    
    # Recent transactions
    st.subheader("📋 Recent Transactions")
    recent_transactions = df.nlargest(10, 'Date')[['Date', 'Amount', 'Description', 'Category']]
    st.dataframe(recent_transactions, use_container_width=True)
    
    # Quick budget status
    if any(budget > 0 for budget in st.session_state.budgets.values()):
        st.subheader("💰 Budget Status")
        actual_spending = df.groupby('Category')['Amount'].sum()
        
        budget_status = []
        for category, budget in st.session_state.budgets.items():
            if budget > 0:
                actual = actual_spending.get(category, 0)
                status = "Over Budget" if actual > budget else "Within Budget"
                budget_status.append({
                    'Category': category,
                    'Budget': budget,
                    'Actual': actual,
                    'Status': status,
                    'Remaining': budget - actual
                })
        
        if budget_status:
            budget_df = pd.DataFrame(budget_status)
            st.dataframe(budget_df, use_container_width=True)
    
    # Download section
    st.subheader("📥 Download Reports")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Generate PDF Report"):
            report_data = create_pdf_report(df, st.session_state.budgets)
            st.success("PDF report generated! (Note: In a real deployment, this would create a downloadable PDF)")
            st.json(report_data)
    
    with col2:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📊 Download CSV",
            data=csv,
            file_name=f"expenses_report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
