"""
PRODUCT RECOMMENDATION SYSTEM
Streamlit Web Application
Company: Miss Misoo Production
Author: Sabin Lamsal
Date: March 2026

This app provides product recommendations using multiple algorithms:
- Popularity Baseline
- Item-Based Collaborative Filtering
- Session-Based Recommendations
- Hybrid Ensemble (Best Performance)
"""


# 1. IMPORTS AND CONFIGURATION

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Product Recommender",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 2. CUSTOM CSS STYLING

def apply_custom_css():
    """Apply custom CSS styling to the app"""
    st.markdown("""
    <style>
        /* Main headers */
        .main-header {
            font-size: 2.8rem;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 1rem;
            font-weight: 700;
            background: linear-gradient(135deg, #4e79a7, #2c3e50);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding: 20px 0;
        }
        
        .sub-header {
            font-size: 1.8rem;
            color: #2c3e50;
            margin: 1.5rem 0 1rem 0;
            font-weight: 600;
            border-left: 6px solid #4e79a7;
            padding-left: 15px;
        }
        
        /* Cards and containers */
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 10px 0;
            border: 1px solid #eaeaea;
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        
        .rec-card {
            background: white;
            padding: 15px 20px;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin: 8px 0;
            border-left: 5px solid #4e79a7;
            display: flex;
            align-items: center;
            transition: all 0.2s ease;
        }
        
        .rec-card:hover {
            background: #f8f9fa;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .rec-rank {
            font-size: 1.4rem;
            font-weight: 700;
            color: #4e79a7;
            min-width: 60px;
            text-align: center;
        }
        
        .rec-id {
            font-size: 1.2rem;
            font-family: 'Courier New', monospace;
            color: #2c3e50;
            font-weight: 500;
        }
        
        .insight-box {
            background: #e8f4f8;
            padding: 20px;
            border-radius: 10px;
            border-left: 6px solid #4e79a7;
            margin: 15px 0;
        }
        
        /* Stats boxes */
        .stat-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        
        .stat-number {
            font-size: 2.2rem;
            font-weight: 700;
        }
        
        .stat-label {
            font-size: 1rem;
            opacity: 0.9;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #4e79a7 0%, #2c3e50 100%);
            color: white;
            font-weight: 600;
            padding: 10px 30px;
            border-radius: 25px;
            border: none;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(78,121,167,0.4);
        }
        
        /* Sidebar */
        .css-1d391kg {
            background: #f8f9fa;
        }
        
        .sidebar-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #2c3e50;
            margin: 20px 0;
            text-align: center;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 30px;
            color: #7f8c8d;
            font-size: 0.9rem;
            border-top: 1px solid #eaeaea;
            margin-top: 50px;
        }
        
        /* Success message */
        .success-msg {
            background: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 8px;
            border-left: 6px solid #28a745;
            margin: 10px 0;
        }
    </style>
    """, unsafe_allow_html=True)


# 3. DATA LOADING FUNCTIONS

@st.cache_resource
def load_models():
    """
    Load all saved models and data from pickle files
    Returns: (pipeline dict, popularity_df DataFrame) or (None, None) if error
    """
    try:
        # Define paths
        model_path = "models/model_pipeline.pkl"
        popularity_path = "models/item_popularity.csv"
        
        # Check if files exist
        if not os.path.exists(model_path):
            st.error(f" Model file not found: {model_path}")
            st.info("Please run Notebook 2 first to generate the models.")
            return None, None
            
        if not os.path.exists(popularity_path):
            st.error(f" Popularity file not found: {popularity_path}")
            return None, None
        
        # Load pipeline
        with open(model_path, 'rb') as f:
            pipeline = pickle.load(f)
        
        # Load popularity data
        popularity_df = pd.read_csv(popularity_path)
        
        # Validate pipeline structure
        required_keys = ['popular_items', 'item_popularity']
        missing_keys = [key for key in required_keys if key not in pipeline]
        
        if missing_keys:
            st.warning(f"⚠️ Pipeline missing some keys: {missing_keys}")
        
        # Ensure all necessary keys exist with defaults
        if 'user_seen' not in pipeline:
            pipeline['user_seen'] = {}
        
        if 'item_next' not in pipeline:
            pipeline['item_next'] = {}
        
        if 'item_similarity' not in pipeline:
            pipeline['item_similarity'] = {}
        
        return pipeline, popularity_df
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def get_user_history(user_id, pipeline):
    """Safely get user's interaction history"""
    try:
        user_seen = pipeline.get('user_seen', {})
        if isinstance(user_seen, dict):
            # Convert string keys to int if needed
            if isinstance(user_id, int) and str(user_id) in user_seen:
                return user_seen[str(user_id)]
            return user_seen.get(user_id, [])
        return []
    except Exception:
        return []

def get_similar_items(item_id, pipeline, max_items=20):
    """Safely get similar items for item-based CF"""
    try:
        item_sim = pipeline.get('item_similarity', {})
        if isinstance(item_sim, dict):
            sim_items = item_sim.get(item_id, {})
            if isinstance(sim_items, dict):
                # Sort by similarity score
                sorted_items = sorted(sim_items.items(), key=lambda x: x[1], reverse=True)
                return [(item, score) for item, score in sorted_items[:max_items]]
        return []
    except Exception:
        return []

def get_item_transitions(item_id, pipeline, max_items=20):
    """Safely get next item probabilities for session-based"""
    try:
        item_next = pipeline.get('item_next', {})
        if isinstance(item_next, dict):
            transitions = item_next.get(item_id, [])
            if isinstance(transitions, list):
                return transitions[:max_items]
        return []
    except Exception:
        return []


# 4. RECOMMENDATION ALGORITHMS

def get_popularity_recommendations(user_id, pipeline, popularity_df, k=10):
    """
    Popularity-based recommendations
    Recommends most popular items the user hasn't seen
    """
    try:
        # Get popular items
        if popularity_df is not None and len(popularity_df) > 0:
            popular_items = popularity_df['itemid'].tolist()
        else:
            # Fallback items
            popular_items = pipeline.get('popular_items', [1001, 1002, 1003, 1004, 1005])
        
        # Get user's seen items
        seen_items = get_user_history(user_id, pipeline)
        seen_set = set(seen_items) if seen_items else set()
        
        # Filter out seen items
        recommendations = []
        for item in popular_items:
            if item not in seen_set:
                recommendations.append(item)
                if len(recommendations) >= k:
                    break
        
        # If not enough, add generic items
        while len(recommendations) < k:
            recommendations.append(f"item_{len(recommendations)+1}")
        
        return recommendations[:k]
    
    except Exception as e:
        st.warning(f"Popularity recommender error: {str(e)}")
        return [f"item_{i}" for i in range(1, k+1)]

def get_itemcf_recommendations(user_id, pipeline, popularity_df, k=10):
    """
    Item-Based Collaborative Filtering
    Recommends items similar to what user has viewed
    """
    try:
        # Get user's seen items
        seen_items = get_user_history(user_id, pipeline)
        
        if not seen_items:
            return get_popularity_recommendations(user_id, pipeline, popularity_df, k)
        
        seen_set = set(seen_items)
        
        # Score candidates based on similarity to seen items
        scores = defaultdict(float)
        
        for seen_item in seen_items[-5:]:  # Use last 5 items
            similar_items = get_similar_items(seen_item, pipeline)
            
            for sim_item, sim_score in similar_items:
                if sim_item not in seen_set:
                    scores[sim_item] += sim_score
        
        if not scores:
            return get_popularity_recommendations(user_id, pipeline, popularity_df, k)
        
        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in ranked[:k]]
    
    except Exception as e:
        st.warning(f"Item-CF error: {str(e)}")
        return get_popularity_recommendations(user_id, pipeline, popularity_df, k)

def get_session_recommendations(user_id, pipeline, popularity_df, k=10):
    """
    Session-Based Recommendations
    Uses sequential patterns and recency weighting
    """
    try:
        # Get user's seen items
        seen_items = get_user_history(user_id, pipeline)
        seen_set = set(seen_items) if seen_items else set()
        
        if not seen_items:
            return get_popularity_recommendations(user_id, pipeline, popularity_df, k)
        
        # Use recent items as session context
        recent_items = seen_items[-5:] if len(seen_items) >= 5 else seen_items
        
        scores = defaultdict(float)
        
        # Apply recency weighting (exponential decay)
        for i, item in enumerate(reversed(recent_items)):
            recency_weight = 0.8 ** i  # Decay factor
            
            transitions = get_item_transitions(item, pipeline)
            
            for next_item, prob in transitions:
                if next_item not in seen_set:
                    scores[next_item] += prob * recency_weight
        
        if not scores:
            return get_popularity_recommendations(user_id, pipeline, popularity_df, k)
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in ranked[:k]]
    
    except Exception as e:
        st.warning(f"Session-based error: {str(e)}")
        return get_popularity_recommendations(user_id, pipeline, popularity_df, k)

def get_hybrid_recommendations(user_id, pipeline, popularity_df, k=10):
    """
    Hybrid Recommendations
    Combines multiple strategies with dynamic weighting based on user history
    """
    try:
        # Get user's seen items
        seen_items = get_user_history(user_id, pipeline)
        n_history = len(seen_items)
        
        # Dynamic weights based on user history length
        if n_history == 0:  # New user
            weights = {
                'popularity': 0.7,
                'session': 0.3,
                'item_cf': 0.0
            }
        elif n_history < 3:  # Few interactions
            weights = {
                'popularity': 0.3,
                'session': 0.5,
                'item_cf': 0.2
            }
        elif n_history < 10:  # Some history
            weights = {
                'popularity': 0.1,
                'session': 0.4,
                'item_cf': 0.5
            }
        else:  # Power user
            weights = {
                'popularity': 0.05,
                'session': 0.35,
                'item_cf': 0.6
            }
        
        n_candidates = k * 5
        seen_set = set(seen_items) if seen_items else set()
        combined_scores = defaultdict(float)
        
        # Get recommendations from each model
        if weights['popularity'] > 0:
            pop_recs = get_popularity_recommendations(user_id, pipeline, popularity_df, n_candidates)
            for rank, item in enumerate(pop_recs):
                if item not in seen_set:
                    score = weights['popularity'] * (n_candidates - rank) / n_candidates
                    combined_scores[item] += score
        
        if weights['session'] > 0:
            sess_recs = get_session_recommendations(user_id, pipeline, popularity_df, n_candidates)
            for rank, item in enumerate(sess_recs):
                if item not in seen_set:
                    score = weights['session'] * (n_candidates - rank) / n_candidates
                    combined_scores[item] += score
        
        if weights['item_cf'] > 0:
            item_recs = get_itemcf_recommendations(user_id, pipeline, popularity_df, n_candidates)
            for rank, item in enumerate(item_recs):
                if item not in seen_set:
                    score = weights['item_cf'] * (n_candidates - rank) / n_candidates
                    combined_scores[item] += score
        
        if not combined_scores:
            return get_popularity_recommendations(user_id, pipeline, popularity_df, k)
        
        # Sort by combined score
        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in ranked[:k]]
    
    except Exception as e:
        st.warning(f"Hybrid recommender error: {str(e)}")
        return get_popularity_recommendations(user_id, pipeline, popularity_df, k)


# 5. UI PAGES

def home_page(pipeline, popularity_df):
    """Home page with overview and statistics"""
    
    st.markdown("<h1 class='main-header'>🛍️ Product Recommender System</h1>", unsafe_allow_html=True)
    
    # Welcome message
    st.markdown("""
    <div class='insight-box'>
        <h3>🌟 Welcome to the Product Recommendation System</h3>
        <p>This intelligent system helps e-commerce platforms suggest relevant products 
        to users based on their browsing behavior, purchase history, and session context.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>📊 Dataset</h3>
            <h2>2.76M</h2>
            <p>Total Events</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3>👥 Users</h3>
            <h2>1.41M</h2>
            <p>Unique Visitors</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h3>📦 Products</h3>
            <h2>235K</h2>
            <p>Unique Items</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='metric-card'>
            <h3>🏆 Best Model</h3>
            <h2>Session</h2>
            <p>MAP@10: 0.0135</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Features section
    st.markdown("<h2 class='sub-header'>✨ Key Features</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>🎯 Multiple Algorithms</h3>
            <ul>
                <li><b>Popularity Baseline</b> - Most viewed items</li>
                <li><b>Item-Based CF</b> - Similar products</li>
                <li><b>Session-Based</b> - Real-time context</li>
                <li><b>Hybrid Ensemble</b> - Best of all worlds</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3>📈 Smart Features</h3>
            <ul>
                <li>Cold-start handling</li>
                <li>Real-time adaptation</li>
                <li>Recency weighting</li>
                <li>Dynamic personalization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start guide
    st.markdown("<h2 class='sub-header'>🚀 Quick Start Guide</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='metric-card'>
        <ol style='font-size: 1.1rem;'>
            <li><b>Navigate to "Recommendations"</b> page from the sidebar</li>
            <li><b>Enter a User ID</b> (try: 1, 100, 1000, or "new_user")</li>
            <li><b>Choose a recommendation model</b></li>
            <li><b>Click "Get Recommendations"</b> to see personalized results</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample popular items
    if popularity_df is not None and len(popularity_df) > 0:
        st.markdown("<h2 class='sub-header'>🔥 Most Popular Items</h2>", unsafe_allow_html=True)
        
        top_items = popularity_df.head(10)
        cols = st.columns(5)
        
        for i, (idx, row) in enumerate(top_items.iterrows()):
            with cols[i % 5]:
                st.markdown(f"""
                <div style='text-align: center; padding: 10px; background: #f8f9fa; 
                            border-radius: 8px; margin: 5px;'>
                    <h4>📦 Product</h4>
                    <h3>{row['itemid']}</h3>
                    <p>{row['interaction_count']} interactions</p>
                </div>
                """, unsafe_allow_html=True)

def recommendations_page(pipeline, popularity_df):
    """Main recommendations page"""
    
    st.markdown("<h1 class='main-header'>🎯 Get Personalized Recommendations</h1>", unsafe_allow_html=True)
    
    # Check if models are loaded
    if pipeline is None or popularity_df is None:
        st.error("Models not loaded properly. Please check the model files.")
        return
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("<div class='sidebar-header'>⚙️ Settings</div>", unsafe_allow_html=True)
        
        # User input
        user_input = st.text_input("Enter User ID:", value="1", key="user_id_input")
        
        # Model selection
        model_options = [
            "Hybrid (Best Overall)",
            "Session-Based (Real-time)",
            "Item-Based CF (Similar Items)",
            "Popularity Baseline",
            "Compare All Models"
        ]
        
        selected_model = st.selectbox(
            "Choose Recommendation Model:",
            model_options,
            index=0
        )
        
        # Number of recommendations
        k = st.slider("Number of Recommendations:", min_value=5, max_value=20, value=10, step=5)
        
        # Get recommendations button
        recommend_btn = st.button("🔍 Get Recommendations", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("""
        <div style='background: #e8f4f8; padding: 10px; border-radius: 8px;'>
            <h4>💡 Pro Tip</h4>
            <p>Try different User IDs:</p>
            <ul>
                <li><b>2</b> - Active user</li>
                <li><b>100</b> - Moderate history</li>
                <li><b>1000</b> - New user</li>
                <li><b>new_user</b> - Cold start</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    if recommend_btn:
        with st.spinner("🔄 Generating recommendations..."):
            time.sleep(1)  # Small delay for better UX
            
            # Process user ID
            try:
                user_id = int(user_input) if user_input.isdigit() else user_input
            except:
                user_id = user_input
            
            # Get user history info
            user_history = get_user_history(user_id, pipeline)
            history_length = len(user_history)
            
            # Display user info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>👤 User</h3>
                    <h2>{user_id}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>📊 History</h3>
                    <h2>{history_length}</h2>
                    <p>interactions</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                user_type = "New User" if history_length == 0 else "Active User" if history_length < 5 else "Power User"
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>🏷️ Type</h3>
                    <h2>{user_type}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Generate recommendations based on selection
            if selected_model == "Popularity Baseline":
                recs = get_popularity_recommendations(user_id, pipeline, popularity_df, k)
                model_name = "Popularity Baseline"
                
                st.markdown(f"<h2 class='sub-header'>📊 {model_name} Recommendations</h2>", unsafe_allow_html=True)
                
                for i, item in enumerate(recs, 1):
                    st.markdown(f"""
                    <div class='rec-card'>
                        <span class='rec-rank'>#{i}</span>
                        <span class='rec-id'>Product {item}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            elif selected_model == "Item-Based CF (Similar Items)":
                recs = get_itemcf_recommendations(user_id, pipeline, popularity_df, k)
                model_name = "Item-Based Collaborative Filtering"
                
                st.markdown(f"<h2 class='sub-header'>🔄 {model_name}</h2>", unsafe_allow_html=True)
                st.markdown("<p><i>Recommended based on items you've viewed</i></p>", unsafe_allow_html=True)
                
                for i, item in enumerate(recs, 1):
                    st.markdown(f"""
                    <div class='rec-card'>
                        <span class='rec-rank'>#{i}</span>
                        <span class='rec-id'>Product {item}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            elif selected_model == "Session-Based (Real-time)":
                recs = get_session_recommendations(user_id, pipeline, popularity_df, k)
                model_name = "Session-Based"
                
                st.markdown(f"<h2 class='sub-header'>⏱️ {model_name} Recommendations</h2>", unsafe_allow_html=True)
                st.markdown("<p><i>Real-time recommendations based on current session</i></p>", unsafe_allow_html=True)
                
                for i, item in enumerate(recs, 1):
                    st.markdown(f"""
                    <div class='rec-card'>
                        <span class='rec-rank'>#{i}</span>
                        <span class='rec-id'>Product {item}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            elif selected_model == "Hybrid (Best Overall)":
                recs = get_hybrid_recommendations(user_id, pipeline, popularity_df, k)
                model_name = "Hybrid Ensemble"
                
                st.markdown(f"<h2 class='sub-header'>🎯 {model_name} (Best Performance)</h2>", unsafe_allow_html=True)
                st.markdown("<p><i>Combines multiple strategies for optimal results</i></p>", unsafe_allow_html=True)
                
                for i, item in enumerate(recs, 1):
                    st.markdown(f"""
                    <div class='rec-card'>
                        <span class='rec-rank'>#{i}</span>
                        <span class='rec-id'>Product {item}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            elif selected_model == "Compare All Models":
                st.markdown("<h2 class='sub-header'>📊 Model Comparison</h2>", unsafe_allow_html=True)
                
                # Get recommendations from all models
                pop_recs = get_popularity_recommendations(user_id, pipeline, popularity_df, k)
                item_recs = get_itemcf_recommendations(user_id, pipeline, popularity_df, k)
                sess_recs = get_session_recommendations(user_id, pipeline, popularity_df, k)
                hybrid_recs = get_hybrid_recommendations(user_id, pipeline, popularity_df, k)
                
                # Create tabs for each model
                tab1, tab2, tab3, tab4 = st.tabs([
                    "🎯 Hybrid (Best)", 
                    "⏱️ Session-Based", 
                    "🔄 Item-Based", 
                    "📊 Popularity"
                ])
                
                with tab1:
                    st.markdown("#### Hybrid Ensemble Recommendations")
                    for i, item in enumerate(hybrid_recs, 1):
                        st.markdown(f"{i}. Product **{item}**")
                
                with tab2:
                    st.markdown("#### Session-Based Recommendations")
                    for i, item in enumerate(sess_recs, 1):
                        st.markdown(f"{i}. Product **{item}**")
                
                with tab3:
                    st.markdown("#### Item-Based CF Recommendations")
                    for i, item in enumerate(item_recs, 1):
                        st.markdown(f"{i}. Product **{item}**")
                
                with tab4:
                    st.markdown("#### Popularity Baseline")
                    for i, item in enumerate(pop_recs, 1):
                        st.markdown(f"{i}. Product **{item}**")
                
                recs = hybrid_recs  # Use hybrid for download
            
            # Download button
            if recs:
                rec_df = pd.DataFrame({
                    'Rank': range(1, len(recs)+1),
                    'Product ID': recs,
                    'Model': selected_model,
                    'User ID': user_id
                })
                
                csv = rec_df.to_csv(index=False)
                
                st.markdown("---")
                st.download_button(
                    label="📥 Download Recommendations as CSV",
                    data=csv,
                    file_name=f"recommendations_user_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

def analysis_page(pipeline, popularity_df):
    """Data analysis and insights page"""
    
    st.markdown("<h1 class='main-header'>📊 Data Analysis & Insights</h1>", unsafe_allow_html=True)
    
    if popularity_df is not None and len(popularity_df) > 0:
        # Key statistics
        st.markdown("<h2 class='sub-header'>📈 Key Statistics</h2>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_items = len(popularity_df)
            st.metric("Total Products", f"{total_items:,}")
        
        with col2:
            avg_interactions = popularity_df['interaction_count'].mean()
            st.metric("Avg Interactions/Product", f"{avg_interactions:.1f}")
        
        with col3:
            cold_items = (popularity_df['interaction_count'] <= 5).sum()
            cold_pct = (cold_items/total_items)*100
            st.metric("Cold Items (≤5)", f"{cold_items:,} ({cold_pct:.1f}%)")
        
        with col4:
            hot_items = (popularity_df['interaction_count'] >= 100).sum()
            hot_pct = (hot_items/total_items)*100
            st.metric("Hot Items (≥100)", f"{hot_items:,} ({hot_pct:.3f}%)")
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Top 20 Most Popular Products")
            
            top20 = popularity_df.head(20)
            fig = px.bar(
                top20,
                x='itemid',
                y='interaction_count',
                title="Product Popularity Distribution",
                labels={'itemid': 'Product ID', 'interaction_count': 'Number of Interactions'},
                color='interaction_count',
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📉 Long Tail Distribution")
            
            # Create bins for distribution
            bins = [0, 1, 5, 10, 20, 50, 100, 500, 1000, 5000]
            labels = ['1', '2-5', '6-10', '11-20', '21-50', '51-100', '101-500', '501-1000', '1000+']
            
            popularity_df['bin'] = pd.cut(popularity_df['interaction_count'], bins=bins, labels=labels)
            bin_counts = popularity_df['bin'].value_counts().sort_index()
            
            fig = px.pie(
                values=bin_counts.values,
                names=bin_counts.index,
                title="Product Distribution by Interaction Count",
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Insights
        st.markdown("<h2 class='sub-header'>💡 Key Insights</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='insight-box'>
            <h4>1. Extreme Long Tail Distribution</h4>
            <p>• <b>31.3%</b> of products have only 1 interaction</p>
            <p>• <b>78.3%</b> of products have ≤10 interactions</p>
            <p>• This indicates a classic e-commerce long tail - most products are rarely interacted with</p>
        </div>
        
        <div class='insight-box'>
            <h4>2. User Behavior Patterns</h4>
            <p>• <b>71.2%</b> of users have only 1 event (cold start is the norm!)</p>
            <p>• <b>91.4%</b> of users have ≤3 events</p>
            <p>• This explains why session-based recommendations outperform user-based CF</p>
        </div>
        
        <div class='insight-box'>
            <h4>3. Business Implications</h4>
            <p>• <b>Cold-start handling</b> is critical - most users have minimal history</p>
            <p>• <b>Session context</b> provides the strongest signal for recommendations</p>
            <p>• <b>Popularity baseline</b> is actually quite effective for new users</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model performance comparison
        st.markdown("<h2 class='sub-header'>📊 Model Performance Comparison</h2>", unsafe_allow_html=True)
        
        performance_data = pd.DataFrame({
            'Model': ['Popularity', 'Item-CF', 'User-CF', 'SVD', 'Session', 'Hybrid'],
            'Precision@10': [0.01109, 0.01174, 0.00892, 0.01086, 0.01350, 0.01293],
            'Recall@10': [0.01124, 0.01189, 0.00923, 0.01101, 0.01372, 0.01308],
            'MAP@10': [0.01109, 0.01174, 0.00892, 0.01086, 0.01350, 0.01293]
        })
        
        fig = px.bar(
            performance_data.melt(id_vars=['Model'], var_name='Metric', value_name='Score'),
            x='Model',
            y='Score',
            color='Metric',
            title="Model Performance Comparison (K=10)",
            barmode='group',
            color_discrete_sequence=['#4e79a7', '#f28e2b', '#59a14f']
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

def about_page():
    """About page with documentation"""
    
    st.markdown("<h1 class='main-header'>ℹ️ About This Project</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='metric-card'>
        <h2>🎯 Project Overview</h2>
        <p>This recommendation system was developed for <b>Miss Misoo Production</b> as part 
        of a Machine Learning Internship project. It uses real e-commerce clickstream data 
        from Retailrocket to provide personalized product recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>📊 Dataset</h3>
            <ul>
                <li><b>Source:</b> Retailrocket Recommender Dataset</li>
                <li><b>Period:</b> 4.5 months (May-Sep 2015)</li>
                <li><b>Events:</b> 2,755,641 interactions</li>
                <li><b>Users:</b> 1,407,580 unique visitors</li>
                <li><b>Products:</b> 235,061 unique items</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3>🤖 Models Implemented</h3>
            <ul>
                <li><b>Popularity Baseline</b> - Simple benchmark</li>
                <li><b>Item-Based CF</b> - Jaccard similarity</li>
                <li><b>User-Based CF</b> - Cosine similarity</li>
                <li><b>Matrix Factorization (SVD)</b> - Latent factors</li>
                <li><b>Session-Based</b> - Sequential patterns 🏆</li>
                <li><b>Hybrid Ensemble</b> - Weighted combination</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='metric-card'>
        <h3>📈 Performance Summary</h3>
    </div>
    """, unsafe_allow_html=True)
    
    perf_df = pd.DataFrame({
        'Model': ['Popularity', 'Item-CF', 'User-CF', 'SVD', 'Session', 'Hybrid'],
        'MAP@10': [0.01109, 0.01174, 0.00892, 0.01086, 0.01350, 0.01293],
        'vs Baseline': ['-', '+5.9%', '-19.6%', '-2.1%', '+21.6%', '+16.6%']
    })
    
    st.dataframe(perf_df, use_container_width=True)
    
    st.markdown("""
    <div class='metric-card'>
        <h3>👨‍💻 Developer</h3>
        <p><b>Sabin Lamsal</b><br>
        Machine Learning Intern<br>
        Miss Misoo Production<br>
        March 2026</p>
    </div>
    
    <div class='footer'>
        © 2026 Miss Misoo Production. All rights reserved.
    </div>
    """, unsafe_allow_html=True)


# 6. MAIN APP


def main():
    """Main application entry point"""
    
    # Apply custom CSS
    apply_custom_css()
    
    # Load models
    pipeline, popularity_df = load_models()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h1 style='color: #4e79a7;'>🛍️ Miss Misoo</h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigate to:",
            ["🏠 Home", "🎯 Recommendations", "📊 Analysis", "ℹ️ About"],
            index=0
        )
        
        st.markdown("---")
        
        # Status indicator
        if pipeline is not None:
            st.success("Models loaded successfully")
        else:
            st.error("Models not loaded")
        
        st.markdown("---")
        st.markdown("© 2026 Miss Misoo Production")
    
    # Route to appropriate page
    if page == "🏠 Home":
        home_page(pipeline, popularity_df)
    elif page == "🎯 Recommendations":
        recommendations_page(pipeline, popularity_df)
    elif page == "📊 Analysis":
        analysis_page(pipeline, popularity_df)
    elif page == "ℹ️ About":
        about_page()


# 7. RUN THE APP


if __name__ == "__main__":
    main()
