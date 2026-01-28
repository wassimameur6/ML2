"""
Churn Prediction AI Agent - Uses RAG + LLM for intelligent retention
"""
import os
import pickle
import json
import smtplib
import uuid
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


@dataclass
class CustomerProfile:
    """Customer data structure"""
    client_num: int
    first_name: str
    last_name: str
    email: str
    phone: str
    age: int
    gender: str
    dependent_count: int
    education_level: str
    marital_status: str
    income_category: str
    card_category: str
    months_on_book: int
    total_relationship_count: int
    months_inactive_12_mon: int
    contacts_count_12_mon: int
    credit_limit: float
    total_revolving_bal: float
    avg_open_to_buy: float
    total_amt_chng_q4_q1: float
    total_trans_amt: float
    total_trans_ct: int
    total_ct_chng_q4_q1: float
    avg_utilization_ratio: float

    @classmethod
    def from_dict(cls, data: Dict) -> 'CustomerProfile':
        return cls(
            client_num=data.get('CLIENTNUM', 0),
            first_name=data.get('First_Name', ''),
            last_name=data.get('Last_Name', ''),
            email=data.get('Email', ''),
            phone=data.get('Phone_Number', ''),
            age=data.get('Customer_Age', 0),
            gender=data.get('Gender', ''),
            dependent_count=data.get('Dependent_count', 0),
            education_level=data.get('Education_Level', ''),
            marital_status=data.get('Marital_Status', ''),
            income_category=data.get('Income_Category', ''),
            card_category=data.get('Card_Category', ''),
            months_on_book=data.get('Months_on_book', 0),
            total_relationship_count=data.get('Total_Relationship_Count', 0),
            months_inactive_12_mon=data.get('Months_Inactive_12_mon', 0),
            contacts_count_12_mon=data.get('Contacts_Count_12_mon', 0),
            credit_limit=data.get('Credit_Limit', 0.0),
            total_revolving_bal=data.get('Total_Revolving_Bal', 0.0),
            avg_open_to_buy=data.get('Avg_Open_To_Buy', 0.0),
            total_amt_chng_q4_q1=data.get('Total_Amt_Chng_Q4_Q1', 0.0),
            total_trans_amt=data.get('Total_Trans_Amt', 0.0),
            total_trans_ct=data.get('Total_Trans_Ct', 0),
            total_ct_chng_q4_q1=data.get('Total_Ct_Chng_Q4_Q1', 0.0),
            avg_utilization_ratio=data.get('Avg_Utilization_Ratio', 0.0)
        )

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    def to_description(self) -> str:
        """Generate natural language description for RAG search"""
        inactive_desc = "inactive" if self.months_inactive_12_mon > 2 else "active"
        spending_desc = "high spender" if self.total_trans_amt > 5000 else "moderate spender" if self.total_trans_amt > 2000 else "low spender"
        utilization_desc = "high credit utilization" if self.avg_utilization_ratio > 0.5 else "low credit utilization"

        return f"""
        Customer with {self.income_category} income, {self.card_category} card holder.
        Been a customer for {self.months_on_book} months.
        Currently {inactive_desc} with {self.months_inactive_12_mon} months of inactivity.
        {spending_desc} with ${self.total_trans_amt:,.0f} in transactions.
        Has {utilization_desc} at {self.avg_utilization_ratio:.0%}.
        Credit limit: ${self.credit_limit:,.0f}.
        Age: {self.age}, {self.marital_status}.
        """


@dataclass
class PredictionResult:
    """Churn prediction result"""
    client_num: int
    churn_probability: float
    churn_risk: str  # low, medium, high
    is_churning: bool

    @classmethod
    def from_probability(cls, client_num: int, probability: float) -> 'PredictionResult':
        if probability < 0.3:
            risk = "low"
        elif probability < 0.6:
            risk = "medium"
        else:
            risk = "high"

        return cls(
            client_num=client_num,
            churn_probability=probability,
            churn_risk=risk,
            is_churning=probability >= 0.5
        )


@dataclass
class OfferRecommendation:
    """Recommended retention offer"""
    offer_id: str
    title: str
    description: str
    offer_type: str
    value: str
    relevance_score: float
    email_subject: str
    email_body: str


class EmailService:
    """Email sending service using SMTP"""

    def __init__(self):
        self.smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', 587))
        self.smtp_user = os.getenv('SMTP_USER', '')
        self.smtp_password = os.getenv('SMTP_PASSWORD', '')
        self.sender_email = os.getenv('SENDER_EMAIL', '')
        self.company_name = os.getenv('COMPANY_NAME', 'Premium Bank')
        self.api_base_url = os.getenv('API_BASE_URL', 'http://localhost:8000')

    def is_configured(self) -> bool:
        """Check if email service is properly configured"""
        return bool(self.smtp_user and self.smtp_password and self.sender_email)

    def personalize_email(self, template: str, customer: CustomerProfile, **kwargs) -> str:
        """Replace template variables with actual values"""
        replacements = {
            '{customer_name}': customer.full_name,
            '{tenure}': str(customer.months_on_book),
            '{company_name}': self.company_name,
            '{fee_amount}': kwargs.get('fee_amount', '95'),
            '{points_away}': kwargs.get('points_away', '5,000'),
            '{new_limit}': kwargs.get('new_limit', f"{customer.credit_limit * 1.25:,.0f}"),
        }

        result = template
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)

        return result

    def send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        customer: CustomerProfile,
        feedback_token: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send a personalized email with optional feedback link"""
        if not self.is_configured():
            return {
                'success': False,
                'error': 'Email service not configured',
                'logged': True,
                'to': to_email,
                'subject': subject
            }

        personalized_subject = self.personalize_email(subject, customer, **kwargs)
        personalized_body = self.personalize_email(body, customer, **kwargs)

        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = personalized_subject
            msg['From'] = f"{self.company_name} <{self.sender_email}>"
            msg['To'] = to_email

            # Plain text version
            plain_text = personalized_body
            if feedback_token:
                plain_text += f"""

────────────────────────────────────────

Interested in this offer? Let us know:

Accept: {self.api_base_url}/feedback/{feedback_token}/accept
No thanks: {self.api_base_url}/feedback/{feedback_token}/decline

{self.company_name}
"""
            text_part = MIMEText(plain_text, 'plain')
            msg.attach(text_part)

            # Professional HTML version
            html_body = self._create_html_email(
                personalized_body,
                customer,
                feedback_token
            )
            html_part = MIMEText(html_body, 'html')
            msg.attach(html_part)

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.sender_email, to_email, msg.as_string())

            return {
                'success': True,
                'to': to_email,
                'subject': personalized_subject,
                'message': 'Email sent successfully',
                'feedback_token': feedback_token
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'to': to_email,
                'subject': personalized_subject
            }

    def _create_html_email(
        self,
        body: str,
        customer: CustomerProfile,
        feedback_token: str = None
    ) -> str:
        """Create professional HTML email with Serfy Bank branding (gold/white theme)"""
        body_html = body.replace('\n', '<br>')

        feedback_section = ""
        if feedback_token:
            feedback_section = f"""
            <tr>
                <td style="padding: 30px 40px; background-color: #FDF8EF; border-top: 1px solid #F5E6C8;">
                    <p style="margin: 0 0 20px 0; font-size: 16px; color: #1a1a1a; text-align: center;">
                        Interested in this offer?
                    </p>
                    <table cellpadding="0" cellspacing="0" border="0" align="center">
                        <tr>
                            <td style="padding-right: 10px;">
                                <a href="{self.api_base_url}/feedback/{feedback_token}/accept"
                                   style="display: inline-block; padding: 14px 32px; background-color: #E5A229;
                                          color: #ffffff; text-decoration: none; font-weight: 600;
                                          font-size: 14px; border-radius: 6px; text-align: center;">
                                    Yes, I'm interested
                                </a>
                            </td>
                            <td style="padding-left: 10px;">
                                <a href="{self.api_base_url}/feedback/{feedback_token}/decline"
                                   style="display: inline-block; padding: 14px 32px; background-color: #ffffff;
                                          color: #1a1a1a; text-decoration: none; font-weight: 600;
                                          font-size: 14px; border-radius: 6px; text-align: center;
                                          border: 2px solid #e5e5e5;">
                                    No, thanks
                                </a>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
            """

        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f5f5f5;">
    <table cellpadding="0" cellspacing="0" border="0" width="100%" style="background-color: #f5f5f5; padding: 40px 20px;">
        <tr>
            <td align="center">
                <table cellpadding="0" cellspacing="0" border="0" width="600" style="max-width: 600px; background-color: #ffffff; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">

                    <!-- Header with Serfy Bank Gold Theme -->
                    <tr>
                        <td style="padding: 32px 40px; background-color: #ffffff; border-bottom: 3px solid #E5A229; border-radius: 12px 12px 0 0; text-align: center;">
                            <h1 style="margin: 0; color: #1a1a1a; font-size: 28px; font-weight: 600;">
                                {self.company_name}
                            </h1>
                            <p style="margin: 8px 0 0 0; color: #888888; font-size: 12px; text-transform: uppercase; letter-spacing: 2px;">
                                Retention Platform
                            </p>
                        </td>
                    </tr>

                    <!-- Gold Accent Bar -->
                    <tr>
                        <td style="background: linear-gradient(90deg, #E5A229 0%, #C78B1F 100%); height: 4px;"></td>
                    </tr>

                    <!-- Body -->
                    <tr>
                        <td style="padding: 40px; color: #1a1a1a; font-size: 15px; line-height: 1.7;">
                            {body_html}
                        </td>
                    </tr>

                    <!-- Feedback Buttons -->
                    {feedback_section}

                    <!-- Footer -->
                    <tr>
                        <td style="padding: 24px 40px; background-color: #1a1a1a; border-radius: 0 0 12px 12px;">
                            <p style="margin: 0; font-size: 12px; color: #888888; text-align: center;">
                                This email was sent to {customer.email}<br>
                                <span style="color: #E5A229;">&copy; 2025 {self.company_name}</span>. All rights reserved.
                            </p>
                        </td>
                    </tr>

                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""


class ChurnAgent:
    """
    AI Agent for churn prediction and retention.

    Flow:
    1. ML Model predicts churn probability
    2. RAG (semantic search) finds relevant offers based on customer profile
    3. LLM ranks and selects the best offer
    4. LLM generates personalized email content
    5. Email sent with feedback link for Human-in-the-Loop
    """

    def __init__(self, artifacts_path: str = None, data_path: str = None):
        base_path = os.path.dirname(os.path.abspath(__file__))  # backend/src/agent
        src_path = os.path.dirname(base_path)  # backend/src

        # artifacts_path: check processors/models first, then artifacts
        if artifacts_path:
            self.artifacts_path = artifacts_path
        elif os.path.exists(os.path.join(src_path, 'processors', 'models')):
            self.artifacts_path = os.path.join(src_path, 'processors', 'models')
        else:
            project_root = os.path.dirname(os.path.dirname(src_path))
            self.artifacts_path = os.path.join(project_root, 'artifacts')

        # data_path: check /app/data (Docker mount) first, then project root
        if data_path:
            self.data_path = data_path
        elif os.path.exists(os.path.join(src_path, 'data')):
            self.data_path = os.path.join(src_path, 'data')
        else:
            project_root = os.path.dirname(os.path.dirname(src_path))
            self.data_path = os.path.join(project_root, 'data')

        self.model = None
        self.preprocessor = None
        self.offers = None
        self.customers_df = None
        self.vectorstore = None

        self.email_service = EmailService()
        self.openai_client = None

        self._load_artifacts()
        self._load_data()
        self._init_openai()
        self._init_vectorstore()

    def _load_artifacts(self):
        """Load trained model and preprocessing artifacts"""
        # Load the new LightGBM pipeline model (SMOTE + LightGBM)
        model_path = os.path.join(self.artifacts_path, 'best_model_final.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("✅ Loaded LightGBM model (best_model_final.pkl)")
        else:
            # Fallback to old model
            with open(os.path.join(self.artifacts_path, 'model.pickle'), 'rb') as f:
                self.model = pickle.load(f)
            print("⚠️ Using legacy Random Forest model (model.pickle)")

        # Build the preprocessor to match the training preprocessing
        self._build_preprocessor()

    def _build_preprocessor(self):
        """
        Build preprocessor matching the training notebook:
        - Feature engineering (5 new ratio features)
        - Fill Unknown values
        - StandardScaler for numerical columns
        - OneHotEncoder(drop='first') for categorical columns
        """
        # Load data to fit the preprocessor
        df = pd.read_csv(os.path.join(self.data_path, 'churn2.csv'))
        df = self._preprocess_dataframe(df)

        X = df.drop(columns=['churn'], errors='ignore')

        # Define column types
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Store column names for later use
        self.num_cols = num_cols
        self.cat_cols = cat_cols

        # Build ColumnTransformer matching the training notebook
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
            ]
        )

        # Fit the preprocessor
        self.preprocessor.fit(X)
        print(f"✅ Preprocessor built: {len(num_cols)} numerical + {len(cat_cols)} categorical features")

    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same preprocessing as the training notebook:
        1. Lowercase column names
        2. Drop unnamed column
        3. Create churn target (if attrition_flag exists)
        4. Fill Unknown values
        5. Feature engineering (ratio features)
        6. Keep only model features
        """
        df = df.copy()

        # Lowercase column names
        df.columns = df.columns.str.lower()

        # Drop unnamed column if exists
        if 'unnamed: 21' in df.columns:
            df = df.drop(columns='unnamed: 21')

        # Create churn target if attrition_flag exists
        if 'attrition_flag' in df.columns:
            df['churn'] = (df['attrition_flag'] == 'Attrited Customer').astype(int)
            df = df.drop(columns='attrition_flag')

        # Fill Unknown values (matching training notebook)
        if 'marital_status' in df.columns:
            df['marital_status'] = df['marital_status'].replace('Unknown', 'Married')
        if 'income_category' in df.columns:
            df['income_category'] = df['income_category'].replace('Unknown', 'Less than $40K')

        # Feature engineering (matching training notebook)
        if 'months_on_book' in df.columns and 'customer_age' in df.columns:
            df['tenure_per_age'] = df['months_on_book'] / (df['customer_age'] * 12)
        if 'avg_utilization_ratio' in df.columns and 'customer_age' in df.columns:
            df['utilisation_per_age'] = df['avg_utilization_ratio'] / df['customer_age']
        if 'credit_limit' in df.columns and 'customer_age' in df.columns:
            df['credit_lim_per_age'] = df['credit_limit'] / df['customer_age']
        if 'total_trans_amt' in df.columns and 'credit_limit' in df.columns:
            df['total_trans_amt_per_credit_lim'] = df['total_trans_amt'] / df['credit_limit']
        if 'total_trans_ct' in df.columns and 'credit_limit' in df.columns:
            df['total_trans_ct_per_credit_lim'] = df['total_trans_ct'] / df['credit_limit']

        # Keep only the columns used in training (drop extra columns like first_name, email, etc.)
        model_features = [
            # Numerical features
            'customer_age', 'dependent_count', 'months_on_book', 'total_relationship_count',
            'months_inactive_12_mon', 'contacts_count_12_mon', 'credit_limit', 'total_revolving_bal',
            'avg_open_to_buy', 'total_amt_chng_q4_q1', 'total_trans_amt', 'total_trans_ct',
            'total_ct_chng_q4_q1', 'avg_utilization_ratio',
            # Engineered features
            'tenure_per_age', 'utilisation_per_age', 'credit_lim_per_age',
            'total_trans_amt_per_credit_lim', 'total_trans_ct_per_credit_lim',
            # Categorical features
            'gender', 'education_level', 'marital_status', 'income_category', 'card_category'
        ]

        # Add churn if present (for fitting purposes)
        if 'churn' in df.columns:
            model_features.append('churn')

        # Select only model features that exist in the dataframe
        available_features = [col for col in model_features if col in df.columns]
        df = df[available_features]

        return df

    def _load_data(self):
        """Load customer data and retention offers"""
        self.customers_df = pd.read_csv(os.path.join(self.data_path, 'churn2.csv'))

        with open(os.path.join(self.data_path, 'retention_offers.json'), 'r') as f:
            self.offers = json.load(f)['offers']

    def _init_openai(self):
        """Initialize OpenAI client"""
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)

    def _init_vectorstore(self):
        """Initialize RAG vector store for semantic offer matching"""
        try:
            from agent.offer_vectorstore import OfferVectorStore
            self.vectorstore = OfferVectorStore(data_path=self.data_path)
            # Ensure offers are indexed
            self.vectorstore.index_offers()
        except Exception as e:
            print(f"Warning: Could not initialize vectorstore: {e}")
            self.vectorstore = None

    def get_customer(self, client_num: int) -> Optional[CustomerProfile]:
        """Get customer by client number"""
        row = self.customers_df[self.customers_df['CLIENTNUM'] == client_num]
        if row.empty:
            return None
        return CustomerProfile.from_dict(row.iloc[0].to_dict())

    def get_all_customers(self) -> List[Dict]:
        """Get all customers as list of dicts"""
        return self.customers_df.to_dict('records')

    def get_high_risk_customers(self, threshold: float = 0.5) -> List[Dict]:
        """Get existing customers with high churn risk (excludes already attrited)"""
        predictions = self.predict_batch(self.customers_df)
        high_risk = [
            {**row, 'churn_probability': pred.churn_probability, 'churn_risk': pred.churn_risk}
            for row, pred in zip(self.customers_df.to_dict('records'), predictions)
            if threshold <= pred.churn_probability < 0.995
            and row.get('Attrition_Flag', '') != 'Attrited Customer'
        ]
        return sorted(high_risk, key=lambda x: x['churn_probability'], reverse=True)

    def _preprocess_features(self, customer_data: Dict) -> np.ndarray:
        """Preprocess customer data for model prediction using the new preprocessor"""
        # Create DataFrame from customer data
        df = pd.DataFrame([customer_data])

        # Apply the same preprocessing as training
        df = self._preprocess_dataframe(df)

        # Remove churn column if present (it's the target, not a feature)
        if 'churn' in df.columns:
            df = df.drop(columns='churn')

        # Transform using the fitted preprocessor
        features = self.preprocessor.transform(df)

        return features

    def predict(self, customer_data: Dict) -> PredictionResult:
        """Predict churn probability for a single customer"""
        features = self._preprocess_features(customer_data)
        probability = self.model.predict_proba(features)[0][1]

        client_num = customer_data.get('CLIENTNUM', 0)
        return PredictionResult.from_probability(client_num, probability)

    def predict_batch(self, customers_df: pd.DataFrame) -> List[PredictionResult]:
        """Predict churn for multiple customers (vectorized for performance)"""
        # Preprocess all customers at once
        df_processed = self._preprocess_dataframe(customers_df.copy())

        # Remove churn column if present
        if 'churn' in df_processed.columns:
            df_processed = df_processed.drop(columns='churn')

        # Transform all at once
        features = self.preprocessor.transform(df_processed)

        # Predict all at once
        probabilities = self.model.predict_proba(features)[:, 1]

        # Create results
        results = []
        client_nums = customers_df['CLIENTNUM'].values if 'CLIENTNUM' in customers_df.columns else range(len(customers_df))
        for client_num, prob in zip(client_nums, probabilities):
            results.append(PredictionResult.from_probability(int(client_num), float(prob)))

        return results

    # =========================================================================
    # RAG: Semantic Search for Offer Matching
    # =========================================================================

    def rag_search_offers(
        self,
        customer: CustomerProfile,
        prediction: PredictionResult,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Use RAG (semantic search) to find the most relevant offers for a customer.
        This searches the vector store based on customer description.
        """
        if not self.vectorstore:
            # Fallback to rule-based matching if vectorstore not available
            return self._rule_based_match(customer, prediction)[:top_k]

        # Create semantic search query from customer profile
        search_query = f"""
        Find retention offer for customer:
        - Income level: {customer.income_category}
        - Card type: {customer.card_category}
        - Customer tenure: {customer.months_on_book} months
        - Churn risk: {prediction.churn_risk}
        - Inactive months: {customer.months_inactive_12_mon}
        - Transaction amount: ${customer.total_trans_amt:,.0f}
        - Credit utilization: {customer.avg_utilization_ratio:.0%}

        Looking for personalized retention offer to prevent this customer from leaving.
        """

        # Semantic search in vector store
        results = self.vectorstore.search_offers(search_query, n_results=top_k)

        return results

    def _rule_based_match(self, customer: CustomerProfile, prediction: PredictionResult) -> List[Dict]:
        """Fallback rule-based matching when RAG is unavailable"""
        matched_offers = []

        for offer in self.offers:
            target = offer['target_profile']
            score = 0

            if customer.income_category in target['income_category']:
                score += 0.3
            if customer.card_category in target['card_category']:
                score += 0.3
            if customer.months_on_book >= target['min_tenure_months']:
                score += 0.2
            if prediction.churn_risk in target['churn_risk']:
                score += 0.2

            if score > 0:
                matched_offers.append({**offer, 'relevance_score': score})

        return sorted(matched_offers, key=lambda x: x['relevance_score'], reverse=True)

    # =========================================================================
    # LLM: Rank and Select Best Offer
    # =========================================================================

    def llm_rank_offers(
        self,
        customer: CustomerProfile,
        prediction: PredictionResult,
        candidate_offers: List[Dict],
        top_k: int = 3
    ) -> List[Dict]:
        """
        Use LLM to intelligently rank and select the best offers for this customer.
        """
        if not self.openai_client or not candidate_offers:
            return candidate_offers[:top_k]

        offers_text = "\n".join([
            f"- {o['offer_id']}: {o['title']} - {o['description']} (Type: {o['offer_type']}, Value: {o['value']})"
            for o in candidate_offers
        ])

        prompt = f"""You are an expert customer retention AI agent. Your task is to rank retention offers for a customer at risk of churning.

CUSTOMER PROFILE:
- Name: {customer.full_name}
- Age: {customer.age}, Gender: {customer.gender}, {customer.marital_status}
- Income: {customer.income_category}
- Card Type: {customer.card_category}
- Customer for: {customer.months_on_book} months
- Monthly Transactions: {customer.total_trans_ct} transactions, ${customer.total_trans_amt:,.0f} total
- Credit Limit: ${customer.credit_limit:,.0f}
- Credit Utilization: {customer.avg_utilization_ratio:.1%}
- Months Inactive: {customer.months_inactive_12_mon}

CHURN ANALYSIS:
- Churn Probability: {prediction.churn_probability:.1%}
- Risk Level: {prediction.churn_risk.upper()}

CANDIDATE OFFERS:
{offers_text}

TASK: Rank these offers from BEST to WORST for this specific customer. Consider:
1. Customer's financial situation (income, credit usage)
2. Their engagement level (activity, transactions)
3. What would most likely retain them
4. The offer's relevance to their profile

Return ONLY the offer IDs in order from best to worst, comma-separated.
Example: OFF003,OFF007,OFF001

YOUR RANKING:"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.3
            )

            # Parse response
            ranked_ids = response.choices[0].message.content.strip().split(',')
            ranked_ids = [id.strip() for id in ranked_ids]

            # Reorder offers based on LLM ranking
            ranked_offers = []
            for offer_id in ranked_ids:
                offer = next((o for o in candidate_offers if o['offer_id'] == offer_id), None)
                if offer:
                    ranked_offers.append(offer)

            # Add any offers not ranked by LLM
            for offer in candidate_offers:
                if offer not in ranked_offers:
                    ranked_offers.append(offer)

            return ranked_offers[:top_k]

        except Exception as e:
            print(f"LLM ranking error: {e}")
            return candidate_offers[:top_k]

    # =========================================================================
    # LLM: Generate Personalized Email Content
    # =========================================================================

    def llm_generate_email(
        self,
        customer: CustomerProfile,
        prediction: PredictionResult,
        offer: Dict
    ) -> Dict[str, str]:
        """
        Use LLM to generate a fully personalized email for this customer.
        """
        if not self.openai_client:
            # Return template-based email if LLM not available
            return {
                'subject': offer['email_subject'],
                'body': offer['email_body']
            }

        company_name = self.email_service.company_name

        prompt = f"""You are a customer retention specialist at {company_name} writing a personalized email.

CUSTOMER:
- Name: {customer.full_name}
- First name: {customer.first_name}
- Been with us: {customer.months_on_book} months
- Income: {customer.income_category}
- Card: {customer.card_category}
- Recent activity: {customer.months_inactive_12_mon} months inactive
- Churn risk: {prediction.churn_risk} ({prediction.churn_probability:.0%})

OFFER TO PRESENT:
- Title: {offer['title']}
- Description: {offer['description']}
- Value: {offer['value']}

TASK: Write a personalized, warm, and compelling retention email. The email should:
1. Address the customer by first name ({customer.first_name})
2. Acknowledge their loyalty ({customer.months_on_book} months)
3. Present the offer naturally and compellingly
4. Create urgency without being pushy
5. Be concise (max 150 words for body)
6. Use proper paragraph breaks (use \\n\\n between paragraphs for spacing)
7. Sign off with "Best regards,\\n\\nThe {company_name} Team"

Respond in this exact JSON format:
{{"subject": "Your email subject here", "body": "Dear [Name],\\n\\nFirst paragraph...\\n\\nSecond paragraph...\\n\\nBest regards,\\n\\nThe {company_name} Team"}}

IMPORTANT: Return ONLY valid JSON, nothing else. Do NOT use placeholders like [Your Name]. Use \\n\\n for paragraph breaks."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.7
            )

            result = response.choices[0].message.content.strip()

            # Clean up the response - remove control characters and fix common issues
            import re
            # Remove control characters except newlines and tabs
            result = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', result)
            # Try to extract JSON if wrapped in markdown code blocks
            if '```' in result:
                json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', result, re.DOTALL)
                if json_match:
                    result = json_match.group(1)

            # Parse JSON response
            email_data = json.loads(result)
            return {
                'subject': email_data.get('subject', offer['email_subject']),
                'body': email_data.get('body', offer['email_body'])
            }

        except Exception as e:
            print(f"LLM email generation error: {e}")
            # Fallback: use template with basic personalization
            body = offer['email_body'].replace('{customer_name}', customer.first_name)
            body = body.replace('{tenure}', str(customer.months_on_book))
            body = body.replace('{company_name}', 'Premium Bank')
            return {
                'subject': offer['email_subject'],
                'body': body
            }

    # =========================================================================
    # Main Agent Flow: RAG + LLM Pipeline
    # =========================================================================

    def recommend_offers(
        self,
        customer: CustomerProfile,
        prediction: PredictionResult,
        top_k: int = 3,
        use_rag: bool = True,
        use_llm_ranking: bool = True
    ) -> List[OfferRecommendation]:
        """
        Complete AI Agent pipeline for offer recommendation:
        1. RAG semantic search for candidate offers
        2. LLM ranking to select best offers
        """
        # Step 1: RAG - Semantic search for relevant offers
        if use_rag and self.vectorstore:
            candidate_offers = self.rag_search_offers(customer, prediction, top_k=10)
        else:
            candidate_offers = self._rule_based_match(customer, prediction)[:10]

        # Step 2: LLM - Rank and select best offers
        if use_llm_ranking and self.openai_client:
            ranked_offers = self.llm_rank_offers(customer, prediction, candidate_offers, top_k=top_k)
        else:
            ranked_offers = candidate_offers[:top_k]

        # Convert to OfferRecommendation objects
        recommendations = []
        for i, offer in enumerate(ranked_offers):
            recommendations.append(OfferRecommendation(
                offer_id=offer['offer_id'],
                title=offer['title'],
                description=offer['description'],
                offer_type=offer['offer_type'],
                value=offer['value'],
                relevance_score=offer.get('relevance_score', 1.0 - (i * 0.1)),
                email_subject=offer['email_subject'],
                email_body=offer['email_body']
            ))

        return recommendations

    def generate_ai_recommendation(
        self,
        customer: CustomerProfile,
        prediction: PredictionResult,
        offers: List[OfferRecommendation]
    ) -> str:
        """Use LLM to generate personalized recommendation explanation"""
        if not self.openai_client:
            return "AI recommendations unavailable - OpenAI not configured"

        prompt = f"""You are a customer retention specialist. Analyze this customer and explain why we chose this retention strategy.

Customer Profile:
- Name: {customer.full_name}
- Age: {customer.age}, Gender: {customer.gender}
- Income: {customer.income_category}
- Card: {customer.card_category}
- Tenure: {customer.months_on_book} months
- Credit Limit: ${customer.credit_limit:,.0f}
- Transaction Amount (last period): ${customer.total_trans_amt:,.0f}
- Months Inactive: {customer.months_inactive_12_mon}

Churn Analysis:
- Churn Probability: {prediction.churn_probability:.1%}
- Risk Level: {prediction.churn_risk.upper()}

Selected Offers (AI-ranked):
{chr(10).join([f"{i+1}. {o.title}: {o.description}" for i, o in enumerate(offers)])}

Provide a brief (2-3 sentences) explanation of why the top offer was selected and how it matches this customer's specific situation."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"AI recommendation error: {str(e)}"

    # =========================================================================
    # Email Sending with Feedback (Human-in-the-Loop)
    # =========================================================================

    def _generate_feedback_token(self, customer: CustomerProfile, offer_id: str) -> str:
        """Generate unique token for feedback tracking"""
        return f"{customer.client_num}_{offer_id}_{uuid.uuid4().hex[:8]}"

    def _store_pending_feedback(self, token: str, customer: CustomerProfile, offer_id: str, prediction: PredictionResult):
        """Store pending feedback for Human-in-the-Loop"""
        feedback_file = os.path.join(self.data_path, 'pending_feedback.json')

        try:
            with open(feedback_file, 'r') as f:
                pending = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pending = {}

        pending[token] = {
            'client_num': customer.client_num,
            'customer_name': customer.full_name,
            'email': customer.email,
            'offer_id': offer_id,
            'churn_probability': prediction.churn_probability,
            'churn_risk': prediction.churn_risk,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending'
        }

        with open(feedback_file, 'w') as f:
            json.dump(pending, f, indent=2)

        return token

    def send_retention_email(
        self,
        customer: CustomerProfile,
        offer: OfferRecommendation,
        prediction: PredictionResult = None,
        use_llm_email: bool = True,
        include_feedback_link: bool = True
    ) -> Dict[str, Any]:
        """
        Send AI-generated retention email with feedback link for Human-in-the-Loop.
        """
        # Generate feedback token for tracking
        feedback_token = None
        if include_feedback_link:
            feedback_token = self._generate_feedback_token(customer, offer.offer_id)
            if prediction:
                self._store_pending_feedback(feedback_token, customer, offer.offer_id, prediction)

        # Use LLM to generate personalized email content
        if use_llm_email and self.openai_client and prediction:
            offer_dict = {
                'title': offer.title,
                'description': offer.description,
                'value': offer.value,
                'email_subject': offer.email_subject,
                'email_body': offer.email_body
            }
            email_content = self.llm_generate_email(customer, prediction, offer_dict)
            subject = email_content['subject']
            body = email_content['body']
        else:
            subject = offer.email_subject
            body = offer.email_body

        # Send email with feedback link
        return self.email_service.send_email(
            to_email=customer.email,
            subject=subject,
            body=body,
            customer=customer,
            feedback_token=feedback_token,
            new_limit=f"{customer.credit_limit * 1.25:,.0f}"
        )

    def run_retention_campaign(
        self,
        customer_ids: List[int] = None,
        risk_threshold: float = 0.5,
        send_emails: bool = False,
        use_rag: bool = True,
        use_llm: bool = True,
        max_customers: int = 5
    ) -> Dict[str, Any]:
        """
        Run AI-powered retention campaign:
        1. Identify high-risk customers
        2. Use RAG to find best offers
        3. Use LLM to rank and personalize
        4. Send emails with feedback links
        """
        if customer_ids:
            customers = [self.get_customer(cid) for cid in customer_ids if self.get_customer(cid)]
        else:
            high_risk = self.get_high_risk_customers(risk_threshold)
            customers = [CustomerProfile.from_dict(c) for c in high_risk[:max_customers]]

        results = {
            'total_customers': len(customers),
            'emails_sent': 0,
            'emails_failed': 0,
            'campaign_details': [],
            'agent_config': {
                'use_rag': use_rag,
                'use_llm': use_llm,
                'vectorstore_available': self.vectorstore is not None,
                'openai_available': self.openai_client is not None
            }
        }

        for customer in customers:
            customer_dict = self.customers_df[
                self.customers_df['CLIENTNUM'] == customer.client_num
            ].iloc[0].to_dict()

            prediction = self.predict(customer_dict)

            # AI Agent: RAG + LLM pipeline
            offers = self.recommend_offers(
                customer, prediction,
                use_rag=use_rag,
                use_llm_ranking=use_llm
            )

            if not offers:
                continue

            top_offer = offers[0]

            detail = {
                'client_num': customer.client_num,
                'name': customer.full_name,
                'email': customer.email,
                'churn_probability': prediction.churn_probability,
                'churn_risk': prediction.churn_risk,
                'recommended_offer': top_offer.title,
                'offer_id': top_offer.offer_id,
                'selection_method': 'RAG+LLM' if (use_rag and use_llm) else 'RAG' if use_rag else 'Rules'
            }

            if send_emails:
                email_result = self.send_retention_email(
                    customer, top_offer, prediction,
                    use_llm_email=use_llm,
                    include_feedback_link=True
                )
                detail['email_sent'] = email_result['success']
                detail['email_error'] = email_result.get('error')
                detail['feedback_token'] = email_result.get('feedback_token')

                if email_result['success']:
                    results['emails_sent'] += 1
                else:
                    results['emails_failed'] += 1

            results['campaign_details'].append(detail)

        return results


if __name__ == "__main__":
    agent = ChurnAgent()

    print("=" * 60)
    print("AI AGENT TEST - RAG + LLM Pipeline")
    print("=" * 60)

    print(f"\nAgent Configuration:")
    print(f"  - VectorStore (RAG): {'✅ Ready' if agent.vectorstore else '❌ Not available'}")
    print(f"  - OpenAI (LLM): {'✅ Connected' if agent.openai_client else '❌ Not configured'}")
    print(f"  - Email Service: {'✅ Configured' if agent.email_service.is_configured() else '❌ Not configured'}")

    high_risk = agent.get_high_risk_customers(threshold=0.7)[:3]
    print(f"\nFound {len(high_risk)} high-risk customers for testing")

    for customer_data in high_risk:
        customer = CustomerProfile.from_dict(customer_data)
        prediction = PredictionResult.from_probability(
            customer.client_num,
            customer_data['churn_probability']
        )

        print(f"\n{'='*40}")
        print(f"Customer: {customer.full_name}")
        print(f"Churn Risk: {prediction.churn_risk} ({prediction.churn_probability:.1%})")

        # Test RAG + LLM pipeline
        offers = agent.recommend_offers(customer, prediction, use_rag=True, use_llm_ranking=True)

        print(f"\nAI-Selected Offers:")
        for i, offer in enumerate(offers, 1):
            print(f"  {i}. {offer.title} (Score: {offer.relevance_score:.0%})")
