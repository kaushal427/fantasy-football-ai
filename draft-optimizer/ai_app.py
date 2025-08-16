import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Import from the core module
from core.ai_draft_assistant import AIDraftAssistant
from core.news_analyzer import NewsAnalyzer
import time

# Page configuration
st.set_page_config(
    page_title="AI Fantasy Football Draft Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'ai_assistant_loaded' not in st.session_state:
    st.session_state.ai_assistant_loaded = False
if 'ai_assistant' not in st.session_state:
    st.session_state.ai_assistant = None

@st.cache_resource
def load_ai_assistant():
    """Load AI assistant and build models"""
    with st.spinner("🤖 Initializing AI Draft Assistant..."):
        ai_assistant = AIDraftAssistant()
        
        # Load all data with AI preprocessing
        ai_assistant.load_all_data()
        
        # Build ML models
        ai_assistant.build_ml_models()
        
        # Classify active players
        ai_assistant.classify_active_players()
        
        return ai_assistant

def main():
    st.title("🤖 AI-Powered Fantasy Football Draft Assistant 2025")
    st.markdown("Advanced machine learning draft optimization with 2025 rookie integration")
    
    # Load AI assistant
    if not st.session_state.ai_assistant_loaded:
        ai_assistant = load_ai_assistant()
        st.session_state.ai_assistant = ai_assistant
        st.session_state.ai_assistant_loaded = True
        st.success("AI Assistant loaded successfully!")
    
    # Sidebar
    st.sidebar.header("AI Settings")
    
    draft_position = st.sidebar.selectbox(
        "Your Draft Position",
        options=list(range(1, 13)),
        help="Select your draft position (1-12)"
    )
    
    # AI Model Information
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 AI Models Active")
    
    if st.session_state.ai_assistant:
        models = st.session_state.ai_assistant.models.keys()
        for model in models:
            st.sidebar.success(f"✅ {model}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "AI Draft Strategy", 
        "Live AI Assistant", 
        "AI Player Rankings", 
        "Rookie Analysis",
        "AI Analytics",
        "Model Insights",
        "AI Settings",
        "League Settings"
    ])
    
    with tab1:
        show_ai_draft_strategy(draft_position)
    
    with tab2:
        show_live_ai_assistant(draft_position)
    
    with tab3:
        show_ai_player_rankings()
    
    with tab4:
        show_rookie_analysis()
    
    with tab5:
        show_ai_analytics()
    
    with tab6:
        show_model_insights()
    
    with tab7:
        show_ai_settings()
    
    with tab8:
        show_league_settings()

def show_ai_draft_strategy(draft_position):
    st.header("🎯 AI-Powered Draft Strategy")
    
    if st.button("🤖 Generate AI Draft Strategy", type="primary"):
        with st.spinner("🧠 AI is analyzing data and creating comprehensive strategy..."):
            # Use saved league settings if available, otherwise use defaults
            if 'league_settings' in st.session_state:
                saved_settings = st.session_state.league_settings
                league_settings = {
                    'teams': saved_settings['num_teams'],
                    'positions': {
                        'QB': saved_settings['starting_qb'],
                        'RB': saved_settings['starting_rb'],
                        'WR': saved_settings['starting_wr'],
                        'TE': saved_settings['starting_te'],
                        'DEF': saved_settings['starting_def'],
                        'K': saved_settings['starting_k']
                    },
                    'ppr': saved_settings['ppr_scoring'],
                    'snake_draft': saved_settings['draft_type'] == 'Snake',
                    'superflex': saved_settings['superflex'],
                    'flex_positions': saved_settings['flex_positions']
                }
            else:
                league_settings = {
                    'teams': 12,
                    'positions': {
                        'QB': 1,
                        'RB': 2,
                        'WR': 2,
                        'TE': 1,
                        'DEF': 1,
                        'K': 1
                    },
                    'ppr': 0.5,
                    'snake_draft': True
                }
            
            strategy = st.session_state.ai_assistant.create_draft_strategy(draft_position, league_settings)
            
            # Display strategy
            st.subheader(f"🤖 AI Draft Strategy for Position #{draft_position}")
            
            # AI Model Information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("AI Models Used", strategy['ai_models_used'])
            with col2:
                st.metric("Total Players Analyzed", len(st.session_state.ai_assistant.current_players))
            with col3:
                st.metric("Strategy Rounds", len(strategy['strategy']))
            
            # Show AI models used
            st.subheader("🧠 AI Models Utilized")
            st.success(f"✅ {strategy['ai_models_used']} AI models used for recommendations")
            
            # Round-by-round strategy
            st.subheader("🎯 AI Round-by-Round Strategy")
            
            for strategy_item in strategy['strategy']:
                with st.expander(f"Round {strategy_item['round']} - AI Recommendation"):
                    pick = strategy_item['pick']
                    
                    # Player info
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f"**{pick['player_name']}**")
                        st.markdown(f"*{pick['position']} - {pick['team']}*")
                        st.markdown(f"**{pick['status']}**")
                    
                    with col2:
                        st.metric("AI Score", f"{pick['score']:.1f}")
                        st.metric("AI Confidence", f"{pick['ai_confidence']:.1%}")
                    
                    with col3:
                        st.metric("Fantasy Points", f"{pick.get('fantasy_points', 0):.1f}")
                        st.metric("Age", pick.get('age', 'N/A'))
                    
                    with col4:
                        st.metric("Last Active", pick['last_active_year'])
                        st.metric("ADP", pick.get('adp', 'N/A'))
                    
                    # Availability information
                    availability = pick.get('availability_probability', 0)
                    if availability > 0.8:
                        st.success(f"✅ High Availability: {availability:.1%}")
                    elif availability > 0.6:
                        st.info(f"🟡 Good Availability: {availability:.1%}")
                    else:
                        st.warning(f"⚠️ Moderate Availability: {availability:.1%}")
                    
                    # AI Reasoning
                    st.markdown("**AI Reasoning:**")
                    st.info(pick['reasoning'])
                    
                    # Alternatives
                    if pick['alternatives']:
                        st.markdown("**🔄 AI Alternatives:**")
                        alt_df = pd.DataFrame(pick['alternatives'])
                        # Filter available columns
                        available_columns = ['player_name', 'position', 'team', 'status', 'score', 'ai_confidence']
                        display_columns = [col for col in available_columns if col in alt_df.columns]
                        
                        st.dataframe(
                            alt_df[display_columns],
                            use_container_width=True
                        )

def show_live_ai_assistant(draft_position):
    st.header("🤖 Live AI Draft Assistant")
    
    # Current round and pick
    col1, col2, col3 = st.columns(3)
    with col1:
        current_round = st.number_input("Current Round", min_value=1, max_value=20, value=1)
    with col2:
        current_pick = st.number_input("Current Pick", min_value=1, max_value=12, value=1)
    with col3:
        is_our_pick = st.checkbox("Is it your pick?", value=True)
    
    if is_our_pick:
        st.info("🤖 AI is analyzing available players and calculating optimal picks...")
        
        # Get available players
        available = st.session_state.ai_assistant.current_players.copy()
        
        # Create draft context
        draft_context = {
            'draft_position': draft_position,
            'round': current_round,
            'team_needs': {
                'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'DEF': 1, 'K': 1
            }
        }
        
        # Get AI recommendation
        pick_id, pick_data = st.session_state.ai_assistant.get_optimal_pick(available, draft_context)
        
        if pick_id:
            # Display AI recommendation
            st.subheader("AI Top Recommendation")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Player", pick_data['player_name'])
                st.markdown(f"**{pick_data['status']}**")
            with col2:
                st.metric("Position", pick_data['position'])
                st.metric("Team", pick_data['team'])
            with col3:
                st.metric("AI Score", f"{pick_data['score']:.1f}")
                st.metric("AI Confidence", f"{pick_data['ai_confidence']:.1%}")
            with col4:
                st.metric("Fantasy Points", f"{pick_data.get('fantasy_points', 0):.1f}")
                st.metric("Age", pick_data.get('age', 'N/A'))
            with col5:
                st.metric("Last Active", pick_data['last_active_year'])
                st.metric("ADP", pick_data.get('adp', 'N/A'))
            
            # Availability information
            availability = pick_data.get('availability_probability', 0)
            if availability > 0.8:
                st.success(f"✅ High Availability: {availability:.1%}")
            elif availability > 0.6:
                st.info(f"🟡 Good Availability: {availability:.1%}")
            else:
                st.warning(f"⚠️ Moderate Availability: {availability:.1%}")
            
            # AI Reasoning
            st.markdown("**🤖 AI Reasoning:**")
            st.info(pick_data['reasoning'])
            
            # Draft button
            if st.button("🤖 Draft This Player (AI Recommended)", type="primary"):
                st.success(f"🤖 AI successfully drafted {pick_data['player_name']}!")
                st.rerun()
            
            # AI Alternatives
            st.subheader("🔄 AI Alternatives")
            if pick_data['alternatives']:
                alt_df = pd.DataFrame(pick_data['alternatives'])
                # Filter available columns
                available_columns = ['player_name', 'position', 'team', 'status', 'score', 'ai_confidence']
                display_columns = [col for col in available_columns if col in alt_df.columns]
                
                st.dataframe(
                    alt_df[display_columns],
                    use_container_width=True
                )
    
    # Available players by position with AI scores
    st.subheader("📋 AI-Scored Available Players")
    
    position = st.selectbox("Select Position", ["QB", "RB", "WR", "TE", "DEF"])
    
    position_players = st.session_state.ai_assistant.current_players[
        st.session_state.ai_assistant.current_players['position'] == position
    ]
    
    if len(position_players) > 0:
        # Calculate AI scores for position players
        draft_context = {'team_needs': {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'DEF': 1}}
        
        scored_players = []
        for _, player in position_players.iterrows():
            score = st.session_state.ai_assistant.calculate_neural_network_score(player, draft_context)
            scored_players.append({
                'player_name': player['player_name'],
                'team': player['team'],
                'status': player['status'],
                'ai_score': score,
                'ai_confidence': player['ai_confidence'],
                'fantasy_points': player['fantasy_points'],
                'age': player['age']
            })
        
        scored_df = pd.DataFrame(scored_players)
        scored_df = scored_df.sort_values('ai_score', ascending=False)
        
        st.dataframe(scored_df, use_container_width=True)
    else:
        st.warning(f"No {position} players available")

def show_ai_player_rankings():
    st.header("📊 AI-Powered Player Rankings")
    
    # Position filter
    position = st.selectbox("Select Position", ["All", "QB", "RB", "WR", "TE", "DEF"])
    
    if position == "All":
        players = st.session_state.ai_assistant.current_players
    else:
        players = st.session_state.ai_assistant.current_players[
            st.session_state.ai_assistant.current_players['position'] == position
        ]
    
    if len(players) > 0:
        # Calculate AI scores
        draft_context = {'team_needs': {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'DEF': 1}}
        
        ai_scores = []
        for _, player in players.iterrows():
            score = st.session_state.ai_assistant.calculate_neural_network_score(player, draft_context)
            ai_scores.append(score)
        
        players_copy = players.copy()
        players_copy['ai_score'] = ai_scores
        players_copy = players_copy.sort_values('ai_score', ascending=False)
        
        # Create interactive chart
        fig = px.scatter(
            players_copy.head(20),
            x='ai_score',
            y='player_name',
            color='status',
            size='ai_confidence',
            hover_data=['team', 'fantasy_points', 'age'],
            title=f"🤖 AI Rankings: Top {position} Players"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        display_columns = ['player_name', 'position', 'team', 'status', 'ai_score', 'ai_confidence', 'fantasy_points', 'age']
        available_columns = [col for col in display_columns if col in players_copy.columns]
        
        st.dataframe(players_copy[available_columns], use_container_width=True)
    else:
        st.warning(f"No {position} players available")

def show_rookie_analysis():
    st.header("⭐ 2025 Rookie Analysis")
    
    if st.session_state.ai_assistant.rookie_projections is not None:
        rookies = st.session_state.ai_assistant.rookie_projections
        
        # Rookie overview
        st.subheader("📊 2025 Rookie Class Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rookies", len(rookies))
        with col2:
            st.metric("Average Athletic Score", f"{rookies['athletic_score'].mean():.2f}")
        with col3:
            st.metric("Average Draft Value", f"{rookies['draft_value'].mean():.2f}")
        
        # Position breakdown
        st.subheader("📈 Rookie Position Breakdown")
        if 'position' in rookies.columns:
            position_counts = rookies['position'].value_counts()
            
            fig = px.pie(
                values=position_counts.values,
                names=position_counts.index,
                title="2025 Rookie Class by Position"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Position data not available for rookies")
        
        # Top rookies by projection
        st.subheader("🏆 Top Rookies by AI Projection")
        
        if 'rookie_projection' in rookies.columns:
            top_rookies = rookies.sort_values('rookie_projection', ascending=False).head(10)
            
            # Use available name column
            name_column = 'player_name' if 'player_name' in rookies.columns else 'Name'
            
            # Check if position column exists for coloring
            position_column = 'position' if 'position' in rookies.columns else 'Position'
            
            if position_column in rookies.columns:
                fig = px.bar(
                    top_rookies,
                    x=name_column,
                    y='rookie_projection',
                    color=position_column,
                    title="Top 10 Rookies by AI Projection Score"
                )
            else:
                fig = px.bar(
                    top_rookies,
                    x=name_column,
                    y='rookie_projection',
                    title="Top 10 Rookies by AI Projection Score"
                )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Rookie projection data not available")
        
        # Rookie data table
        st.subheader("📋 Complete Rookie Analysis")
        available_columns = ['player_name', 'Name', 'position', 'Position', 'team', 'Drafted By', 'athletic_score', 'draft_value', 'rookie_projection']
        display_columns = [col for col in available_columns if col in rookies.columns]
        
        if display_columns:
            rookie_display = rookies[display_columns].copy()
            if 'rookie_projection' in rookie_display.columns:
                rookie_display = rookie_display.sort_values('rookie_projection', ascending=False)
            
            st.dataframe(rookie_display, use_container_width=True)
        else:
            st.info("No rookie data available for display")
        
        # Athletic analysis
        st.subheader("🏃‍♂️ Athletic Analysis")
        
        # Check if athletic data is available
        athletic_columns = ['forty_yard', 'vertical_jump', 'position']
        available_athletic = [col for col in athletic_columns if col in rookies.columns]
        
        if len(available_athletic) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                # 40-yard dash analysis
                if 'forty_yard' in rookies.columns:
                    if 'position' in rookies.columns:
                        fig = px.histogram(
                            rookies,
                            x='forty_yard',
                            color='position',
                            title="40-Yard Dash Distribution by Position"
                        )
                    else:
                        fig = px.histogram(
                            rookies,
                            x='forty_yard',
                            title="40-Yard Dash Distribution"
                        )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Vertical jump analysis
                if 'vertical_jump' in rookies.columns:
                    if 'position' in rookies.columns:
                        fig = px.histogram(
                            rookies,
                            x='vertical_jump',
                            color='position',
                            title="Vertical Jump Distribution by Position"
                        )
                    else:
                        fig = px.histogram(
                            rookies,
                            x='vertical_jump',
                            title="Vertical Jump Distribution"
                        )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Athletic data not available for analysis")
    else:
        st.warning("No rookie data available")

def show_ai_analytics():
    st.header("🔍 AI Analytics")
    
    # Player status distribution
    st.subheader("📊 Player Status Distribution")
    
    current_players = st.session_state.ai_assistant.current_players
    status_counts = current_players['status'].value_counts()
    
    fig = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title="Player Distribution by AI Classification"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # AI confidence distribution
    st.subheader("🤖 AI Confidence Distribution")
    
    fig = px.histogram(
        current_players,
        x='ai_confidence',
        color='status',
        title="AI Confidence Scores by Player Status"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Position analysis
    st.subheader("📈 Position Analysis")
    
    position_analysis = current_players.groupby('position').agg({
        'ai_confidence': 'mean',
        'fantasy_points': 'mean',
        'age': 'mean'
    }).round(2)
    
    st.dataframe(position_analysis, use_container_width=True)
    
    # Team analysis
    st.subheader("🏈 Team Analysis")
    
    team_analysis = current_players.groupby('team').agg({
        'ai_confidence': 'mean',
        'fantasy_points': 'mean'
    }).round(2)
    
    team_analysis = team_analysis.sort_values('fantasy_points', ascending=False)
    
    fig = px.bar(
        team_analysis.head(15),
        x=team_analysis.head(15).index,
        y='fantasy_points',
        title="Average Fantasy Points by Team (Top 15)"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_model_insights():
    st.header("📈 AI Model Insights")
    
    ai_assistant = st.session_state.ai_assistant
    
    # Feature importance
    if 'veteran' in ai_assistant.feature_importance:
        st.subheader("🔍 Veteran Model Feature Importance")
        
        feature_importance = ai_assistant.feature_importance['veteran']
        feature_df = pd.DataFrame([
            {'feature': feature, 'importance': importance}
            for feature, importance in feature_importance.items()
        ])
        feature_df = feature_df.sort_values('importance', ascending=False)
        
        fig = px.bar(
            feature_df,
            x='feature',
            y='importance',
            title="Feature Importance in Veteran Performance Model"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Model performance metrics
    st.subheader("📊 Model Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Veteran Model", "Active" if 'veteran_performance' in ai_assistant.models else "Inactive")
    
    with col2:
        st.metric("Rookie Model", "Active" if ai_assistant.rookie_projections is not None else "Inactive")
    
    with col3:
        st.metric("Injury Model", "Active" if 'injury_risk' in ai_assistant.models else "Inactive")
    
    # Data quality metrics
    st.subheader("📋 Data Quality Metrics")
    
    if ai_assistant.yearly_offense is not None:
        data_quality = {
            'Total Records': len(ai_assistant.yearly_offense),
            '2024 Players': len(ai_assistant.yearly_offense[ai_assistant.yearly_offense['season'] == 2024]),
            '2023 Players': len(ai_assistant.yearly_offense[ai_assistant.yearly_offense['season'] == 2023]),
            'Rookies': len(ai_assistant.rookie_projections) if ai_assistant.rookie_projections is not None else 0
        }
        
        quality_df = pd.DataFrame([
            {'Metric': metric, 'Count': count}
            for metric, count in data_quality.items()
        ])
        
        st.dataframe(quality_df, use_container_width=True)

def show_ai_settings():
    st.header("⚙️ AI Settings")
    
    st.subheader("🤖 AI Model Configuration")
    
    # Model parameters
    st.markdown("**Current AI Models:**")
    
    ai_assistant = st.session_state.ai_assistant
    
    if 'veteran_performance' in ai_assistant.models:
        st.success("✅ Veteran Performance Model (Random Forest)")
        st.markdown("- Uses historical performance data")
        st.markdown("- Predicts future fantasy points")
        st.markdown("- Considers age, experience, and efficiency metrics")
    
    if ai_assistant.rookie_projections is not None:
        st.success("✅ Rookie Projection Model")
        st.markdown("- Combines combine metrics and draft position")
        st.markdown("- Athletic score calculation")
        st.markdown("- Position-specific adjustments")
    
    if 'injury_risk' in ai_assistant.models:
        st.success("✅ Injury Risk Model (Gradient Boosting)")
        st.markdown("- Predicts injury likelihood")
        st.markdown("- Based on age and usage patterns")
    
    # Scoring weights
    st.subheader("⚖️ Scoring Weights")
    
    st.markdown("**Neural Network Scoring Layers:**")
    st.markdown("1. **Base Performance** - Historical stats and projections")
    st.markdown("2. **Context Factors** - Position scarcity and team needs")
    st.markdown("3. **AI Enhancements** - Consistency and upside predictions")
    st.markdown("4. **Meta Factors** - Market inefficiencies and news sentiment")
    
    # Active player bonuses
    st.subheader("🎯 Active Player Bonuses")
    
    bonus_data = {
        'Status': ['🟢 2024 NFL Veteran', '🟡 2023 NFL Veteran', '⭐ 2025 Rookie'],
        'Bonus': [50, 25, 30],
        'Description': ['Currently active', 'Recently active', 'High upside potential']
    }
    
    bonus_df = pd.DataFrame(bonus_data)
    st.dataframe(bonus_df, use_container_width=True)

def show_league_settings():
    st.header("📋 League Settings & Draft Strategy Guide")
    
    # League Configuration
    st.subheader("🏈 League Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_teams = st.selectbox("Number of Teams", [8, 10, 12, 14, 16], index=2)
        draft_type = st.selectbox("Draft Type", ["Snake", "Auction"], index=0)
        ppr_scoring = st.selectbox("PPR Scoring", [0, 0.5, 1.0], index=1)
    
    with col2:
        starting_qb = st.number_input("Starting QBs", min_value=1, max_value=2, value=1)
        starting_rb = st.number_input("Starting RBs", min_value=1, max_value=3, value=2)
        starting_wr = st.number_input("Starting WRs", min_value=1, max_value=4, value=2)
        starting_te = st.number_input("Starting TEs", min_value=1, max_value=2, value=1)
    
    # Position Requirements
    st.subheader("📊 Position Requirements")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        flex_positions = st.number_input("Flex Positions", min_value=0, max_value=3, value=0)
        superflex = st.checkbox("Superflex League", value=False)
    
    with col2:
        starting_def = st.number_input("Starting DEF", min_value=0, max_value=2, value=1)
        starting_k = st.number_input("Starting K", min_value=0, max_value=2, value=1)
    
    with col3:
        bench_size = st.number_input("Bench Size", min_value=3, max_value=10, value=6)
        ir_slots = st.number_input("IR Slots", min_value=0, max_value=5, value=1)
    
    # Draft Strategy Guide
    st.subheader("📚 Value-Based Drafting Strategy Guide")
    
    # Core Philosophy
    st.markdown("### 🎯 Core Philosophy: Wait on QB, Prioritize Skill Positions")
    st.markdown("""
    The foundation of successful fantasy drafting is understanding positional scarcity and value. 
    Quarterbacks should generally be avoided in early rounds because the position has less week-to-week 
    variance and deeper talent pools compared to running backs and wide receivers.
    """)
    
    # Scoring Format Considerations
    st.markdown("### 📊 Scoring Format Considerations")
    
    if ppr_scoring == 0.5:
        st.markdown("**0.5 PPR Leagues**")
        st.markdown("""
        - **Running backs gain significant value** due to their dual rushing/receiving threat
        - Target RBs who catch 40+ passes per season (CMC, Ekeler, Kamara types)
        - Prioritize RBs in rounds 1-3, especially those with goal-line and passing down work
        - Look for "three-down backs" who won't lose snaps in obvious passing situations
        """)
    elif ppr_scoring == 1.0:
        st.markdown("**Full PPR Leagues**")
        st.markdown("""
        - **Wide receivers see the biggest boost** from reception scoring
        - Slot receivers and target hogs become more valuable (think Cooper Kupp, Davante Adams)
        - Consider WR-heavy early rounds (WR-WR or RB-WR-WR starts)
        - Target receivers with 120+ target upside in rounds 1-4
        """)
    else:
        st.markdown("**Standard Scoring**")
        st.markdown("""
        - **Running backs are king** due to touchdown dependency
        - Target goal-line backs and high-touchdown upside players
        - WRs lose significant value without reception points
        - Consider RB-heavy early rounds (RB-RB or RB-RB-WR starts)
        """)
    
    # Round-by-Round Strategy
    st.markdown("### 🎯 Round-by-Round Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Rounds 1-3: Elite Skill Position Players**")
        st.markdown("""
        - **Focus exclusively on RB1s and WR1s** with weekly ceiling potential
        - Avoid QBs entirely - even elite QBs like Josh Allen or Lamar Jackson
        - Look for players with 300+ fantasy point upside
        - Consider positional scarcity: there are only ~12 true RB1s vs ~24 startable WRs
        """)
        
        st.markdown("**Rounds 4-6: Fill Starting Lineup Needs**")
        st.markdown("""
        - Target your RB2/WR2 depending on early round strategy
        - Consider elite tight ends (Kelce, Andrews) if available at value
        - Still avoid QBs unless in superflex/2QB leagues
        - Look for players with safe floors and weekly starter potential
        """)
    
    with col2:
        st.markdown("**Rounds 7-10: Depth and Upside Swings**")
        st.markdown("""
        - **First acceptable QB range** - target QBs with rushing upside (Hurts, Richardson, Daniels)
        - Add depth at RB/WR with handcuffs and breakout candidates
        - Consider streaming defenses vs drafting early
        - Target players in ascending offenses or with expanded roles
        """)
        
        st.markdown("**Rounds 11-15: Late-Round Value and Sleepers**")
        st.markdown("""
        - **Primary QB drafting territory** - plenty of QB1 upside remains
        - Rookie wide receivers in good situations
        - Injury comeback players at discount prices
        - Defense/kicker if you don't plan to stream
        """)
    
    # Why Wait on QB?
    st.markdown("### 🤔 Why Wait on QB?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Positional Depth**")
        st.markdown("""
        - 15-20 QBs typically finish as weekly starters
        - Only 12 RBs and 24 WRs provide consistent RB1/WR1 value
        - QB scoring is more predictable and less volatile week-to-week
        """)
        
        st.markdown("**Opportunity Cost**")
        st.markdown("""
        - Drafting Mahomes in Round 2 costs you a potential 1,500-yard receiver
        - Late-round QBs like Geno Smith or Baker Mayfield often outperform ADP
        - Streaming QBs based on matchups can be as effective as rostering an elite QB
        """)
    
    with col2:
        st.markdown("**Value Examples**")
        st.markdown("""
        - 2023: Tua (Round 10 ADP) outscored Aaron Rodgers (Round 4 ADP)
        - Rookie QBs often provide massive value (see: Stroud, Richardson development)
        - 2024: Jayden Daniels, Caleb Williams could be late-round steals
        """)
    
    # Draft Day Execution Tips
    st.markdown("### 📋 Draft Day Execution Tips")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Pre-Draft Preparation**")
        st.markdown("""
        - Create tiered rankings for each position
        - Identify 3-4 target players per round
        - Research strength of schedule for Weeks 1-4 and fantasy playoffs
        - Know your league's starting requirements and bench size
        """)
        
        st.markdown("**During the Draft**")
        st.markdown("""
        - **Be flexible with your strategy** - don't force picks if value isn't there
        - Monitor other teams' positional needs to predict runs
        - Consider bye weeks when choosing between similar players
        - Don't panic if others draft QBs early - stay disciplined
        """)
    
    with col2:
        st.markdown("**Late Draft Adjustments**")
        st.markdown("""
        - If 8+ QBs are gone by Round 8, consider moving up your QB selection
        - In deep leagues (14+ teams), slight QB priority increase may be necessary
        - Always prioritize best available skill position player over positional need
        """)
        
        st.markdown("**Common Mistakes to Avoid**")
        st.markdown("""
        - **Drafting a QB before Round 7** in standard formats
        - Taking a defense or kicker before the final two rounds
        - Reaching for players based on name recognition over current situation
        - Ignoring target share and snap count projections
        - Drafting too many players from the same NFL team
        """)
    
    # League-Specific Adjustments
    st.markdown("### 🏆 League-Specific Adjustments")
    
    if num_teams == 10:
        st.markdown("**10-Team Leagues**")
        st.markdown("""
        - Even more QB depth available, can wait until Round 10+
        - Focus heavily on ceiling plays since many good players will be available on waivers
        - Consider more aggressive early-round strategies
        """)
    elif num_teams >= 14:
        st.markdown("**14+ Team Leagues**")
        st.markdown("""
        - Slightly earlier QB consideration (Round 8-9 range)
        - Handcuff your RBs more aggressively
        - Target players with clear paths to expanded roles
        - Depth becomes more important than ceiling
        """)
    else:
        st.markdown("**12-Team Leagues**")
        st.markdown("""
        - Standard QB waiting strategy applies
        - Balanced approach between ceiling and floor
        - Moderate handcuff consideration
        """)
    
    # Save Settings Button
    if st.button("💾 Save League Settings", type="primary"):
        league_settings = {
            'num_teams': num_teams,
            'draft_type': draft_type,
            'ppr_scoring': ppr_scoring,
            'starting_qb': starting_qb,
            'starting_rb': starting_rb,
            'starting_wr': starting_wr,
            'starting_te': starting_te,
            'flex_positions': flex_positions,
            'superflex': superflex,
            'starting_def': starting_def,
            'starting_k': starting_k,
            'bench_size': bench_size,
            'ir_slots': ir_slots
        }
        
        st.session_state.league_settings = league_settings
        st.success("✅ League settings saved! These will be used for AI recommendations.")

if __name__ == "__main__":
    main()
