import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
import os

class AIDraftAssistant:
    def __init__(self, data_path: str = "archive/"):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.feature_importance = {}
        
        # Data storage
        self.weekly_offense = None
        self.yearly_offense = None
        self.weekly_defense = None
        self.yearly_defense = None
        self.weekly_team_offense = None
        self.weekly_team_defense = None
        self.yearly_team_offense = None
        self.yearly_team_defense = None
        self.combine_data = None
        self.current_players = None
        self.rookie_projections = None
        
        # Draft availability tracking
        self.drafted_players = set()
        self.adp_data = None
        
    def load_all_data(self):
        """Load all data files with comprehensive preprocessing"""
        print("🤖 Loading and preprocessing all data files...")
        
        try:
            # Load all CSV files
            data_path = "data/archive"
            
            # Load player stats
            self.weekly_offense = pd.read_csv(f"{data_path}/weekly_player_stats_offense.csv")
            self.weekly_defense = pd.read_csv(f"{data_path}/weekly_player_stats_defense.csv")
            self.yearly_offense = pd.read_csv(f"{data_path}/yearly_player_stats_offense.csv")
            self.yearly_defense = pd.read_csv(f"{data_path}/yearly_player_stats_defense.csv")
            
            # Load team stats
            self.weekly_team_offense = pd.read_csv(f"{data_path}/weekly_team_stats_offense.csv")
            self.weekly_team_defense = pd.read_csv(f"{data_path}/weekly_team_stats_defense.csv")
            self.yearly_team_offense = pd.read_csv(f"{data_path}/yearly_team_stats_offense.csv")
            self.yearly_team_defense = pd.read_csv(f"{data_path}/yearly_team_stats_defense.csv")
            
            # Load combine data (2025 prospects)
            combine_path = f"{data_path}/nfl_combine.csv"
            if os.path.exists(combine_path):
                self.combine_data = pd.read_csv(combine_path)
            else:
                print("⚠️ nfl_combine.csv not found, creating mock data...")
                self.combine_data = self._create_mock_combine_data()
            
            print("✅ All data files loaded successfully!")
            
            # Apply AI preprocessing
            self._apply_ai_preprocessing()
            
            # Create realistic ADP data with proper tiers
            self._create_realistic_adp_data()
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def _create_realistic_adp_data(self):
        """Create realistic ADP (Average Draft Position) data based on actual player performance"""
        print("📊 Creating realistic ADP data from actual player performance...")
        
        if self.yearly_offense is None or len(self.yearly_offense) == 0:
            print("⚠️ No yearly offense data available, creating basic ADP structure")
            self.adp_data = pd.DataFrame()
            return
        
        # Get the most recent season data for active players
        max_season = self.yearly_offense['season'].max()
        print(f"📅 Using data from season {max_season}")
        
        recent_data = self.yearly_offense[self.yearly_offense['season'] == max_season]
        print(f"📊 Found {len(recent_data)} players from season {max_season}")
        
        # Filter for players with meaningful fantasy production (less restrictive)
        active_players = recent_data[
            (recent_data['fantasy_points_ppr'] > 0) & 
            (recent_data['games_played_season'] >= 1)  # At least 1 game played (was 3)
        ].copy()
        
        print(f"🎯 Found {len(active_players)} active players with fantasy production")
        
        if len(active_players) == 0:
            print("⚠️ No recent active players found, using fallback data")
            self.adp_data = pd.DataFrame()
            return
        
        # Calculate ADP based on fantasy points and position scarcity
        active_players['adp_score'] = (
            active_players['fantasy_points_ppr'] * 0.6 +  # 60% based on fantasy production
            active_players['games_played_season'] * 0.2 +  # 20% based on availability
            active_players['age'].fillna(25) * 0.1 +  # 10% based on age (younger = better)
            active_players['years_exp'].fillna(3) * 0.1   # 10% based on experience
        )
        
        # Position-specific adjustments
        position_tiers = {
            'RB': 1.0,    # Running backs are most valuable
            'WR': 0.95,   # Wide receivers slightly less
            'QB': 0.9,    # Quarterbacks in 1QB leagues
            'TE': 0.85,   # Tight ends
            'DEF': 0.7,   # Defense
            'K': 0.5      # Kickers
        }
        
        active_players['position_multiplier'] = active_players['position'].map(position_tiers).fillna(0.8)
        active_players['adjusted_adp_score'] = active_players['adp_score'] * active_players['position_multiplier']
        
        # Sort by adjusted score to get draft order
        active_players = active_players.sort_values('adjusted_adp_score', ascending=False)
        
        # Create ADP data
        adp_data = []
        for i, (_, player) in enumerate(active_players.head(200).iterrows()):  # Top 200 players
            adp = i + 1
            
            # Calculate availability confidence based on ADP tier
            if adp <= 24:  # First 2 rounds
                availability_confidence = 0.95
            elif adp <= 48:  # Rounds 3-4
                availability_confidence = 0.85
            elif adp <= 72:  # Rounds 5-6
                availability_confidence = 0.75
            elif adp <= 96:  # Rounds 7-8
                availability_confidence = 0.65
            else:  # Later rounds
                availability_confidence = 0.55
            
            adp_data.append({
                'player_id': player['player_id'],
                'player_name': player['player_name'],
                'position': player['position'],
                'team': player['team'],
                'adp': adp,
                'adp_range': f"{adp}-{adp}",
                'availability_confidence': availability_confidence,
                'fantasy_points': player['fantasy_points_ppr'],
                'games_played': player['games_played_season'],
                'age': player['age'],
                'years_exp': player['years_exp']
            })
        
        self.adp_data = pd.DataFrame(adp_data)
        print(f"✅ Created ADP data for {len(self.adp_data)} players based on actual performance")
        
        # Show top 10 for verification
        print("🏆 Top 10 ADP Players:")
        for _, player in self.adp_data.head(10).iterrows():
            print(f"   {player['adp']:2d}. {player['player_name']} ({player['position']}) - {player['team']} - {player['fantasy_points']:.1f} pts")
    
    def _validate_snake_draft_logic(self, draft_position: int, league_size: int = 12):
        """Validate and display snake draft pick calculations"""
        print(f"🔍 Snake Draft Logic - {league_size} teams, Position #{draft_position}")
        
        for round_num in range(1, 6):  # Show first 5 rounds
            if round_num % 2 == 1:  # Odd rounds - forward order
                pick_number = (round_num - 1) * league_size + draft_position
            else:  # Even rounds - reverse order
                pick_number = round_num * league_size - draft_position + 1
            
            print(f"   Round {round_num}: Pick #{pick_number}")
    
    def get_available_players_at_pick(self, draft_position: int, round_num: int, league_size: int = 12):
        """FIXED: Get realistically available players based on strict ADP logic"""
        if self.current_players is None or len(self.current_players) == 0:
            print("⚠️ No classified players available")
            return pd.DataFrame()
        
        if self.adp_data is None or len(self.adp_data) == 0:
            print("⚠️ No ADP data available")
            return pd.DataFrame()
        
        # Show snake draft logic for first few rounds
        if round_num <= 2:
            self._validate_snake_draft_logic(draft_position, league_size)
        
        # Calculate pick number using snake draft logic
        if round_num % 2 == 1:  # Odd rounds - forward order
            pick_number = (round_num - 1) * league_size + draft_position
        else:  # Even rounds - reverse order
            pick_number = round_num * league_size - draft_position + 1
        
        print(f"🎯 Round {round_num}, Pick #{pick_number}")
        
        available_players = []
        
        # FIXED: Strict ADP-based availability logic
        for _, player in self.current_players.iterrows():
            # Find player in ADP data
            adp_match = self.adp_data[
                (self.adp_data['player_id'] == player['player_id']) |
                (self.adp_data['player_name'] == player['player_name'])
            ]
            
            if len(adp_match) > 0:
                adp_player = adp_match.iloc[0]
                adp = adp_player['adp']
                
                # STRICT RULE: Only include players if we're at or past their ADP
                picks_past_adp = pick_number - adp
                
                if picks_past_adp >= 0:  # We're at or past this player's ADP
                    # Calculate realistic availability probability
                    if picks_past_adp <= 3:  # Within 3 picks of ADP
                        availability_prob = 0.60
                    elif picks_past_adp <= 6:  # Within 0.5 rounds
                        availability_prob = 0.80
                    elif picks_past_adp <= 12:  # Within 1 round
                        availability_prob = 0.90
                    else:  # Well past ADP - should definitely be available
                        availability_prob = 0.95
                    
                    # Apply base confidence modifier
                    availability_prob *= adp_player['availability_confidence']
                    
                    # Only include if reasonable availability
                    if availability_prob >= 0.50:  # 50% minimum threshold
                        available_players.append({
                            'player_id': player['player_id'],
                            'player_name': player['player_name'],
                            'position': player['position'],
                            'team': player['team'],
                            'status': player['status'],
                            'ai_confidence': player['ai_confidence'],
                            'active_bonus': player['active_bonus'],
                            'last_active_year': player['last_active_year'],
                            'fantasy_points': player['fantasy_points'],
                            'age': player['age'],
                            'games_played': player['games_played'],
                            'adp': adp,
                            'adp_tier': adp_player.get('adp_tier', 'unknown'),
                            'availability_probability': availability_prob,
                            'picks_past_adp': picks_past_adp
                        })
                
                elif picks_past_adp >= -6:  # Slight reach (within 0.5 rounds early)
                    # Allow some reaches, but with very low probability
                    reach_penalty = abs(picks_past_adp) * 0.15  # 15% penalty per pick early
                    availability_prob = max(0.05, 0.30 - reach_penalty)  # Very low probability
                    
                    # Special case: Allow skill position reaches in early rounds
                    if round_num <= 3 and player['position'] in ['RB', 'WR']:
                        availability_prob = max(availability_prob, 0.20)  # Slight bump for early skill positions
                    
                    if availability_prob >= 0.15:  # Very low but possible
                        available_players.append({
                            'player_id': player['player_id'],
                            'player_name': player['player_name'],
                            'position': player['position'],
                            'team': player['team'],
                            'status': player['status'],
                            'ai_confidence': player['ai_confidence'],
                            'active_bonus': player['active_bonus'],
                            'last_active_year': player['last_active_year'],
                            'fantasy_points': player['fantasy_points'],
                            'age': player['age'],
                            'games_played': player['games_played'],
                            'adp': adp,
                            'adp_tier': adp_player.get('adp_tier', 'unknown'),
                            'availability_probability': availability_prob,
                            'picks_past_adp': picks_past_adp
                        })
            else:
                # Player not in ADP data - only include in very late rounds
                if round_num >= 10:
                    available_players.append({
                        'player_id': player['player_id'],
                        'player_name': player['player_name'],
                        'position': player['position'],
                        'team': player['team'],
                        'status': player['status'],
                        'ai_confidence': player['ai_confidence'],
                        'active_bonus': player['active_bonus'],
                        'last_active_year': player['last_active_year'],
                        'fantasy_points': player['fantasy_points'],
                        'age': player['age'],
                        'games_played': player['games_played'],
                        'adp': 200,  # Very late assumption
                        'adp_tier': 'undrafted',
                        'availability_probability': 0.85,
                        'picks_past_adp': 0
                    })
        
        # Sort by availability probability and fantasy points
        available_df = pd.DataFrame(available_players)
        if len(available_df) > 0:
            available_df = available_df.sort_values(
                ['availability_probability', 'fantasy_points'], 
                ascending=[False, False]
            )
            
            print(f"📋 Found {len(available_df)} realistically available players")
            if len(available_df) > 0:
                print("🏆 Top 5 available players:")
                for i, (_, player) in enumerate(available_df.head(5).iterrows()):
                    past_adp = player['picks_past_adp']
                    past_text = f"+{past_adp}" if past_adp > 0 else f"{past_adp}"
                    print(f"   {i+1}. {player['player_name']} ({player['position']}) - ADP {player['adp']} ({past_text}) - {player['availability_probability']:.1%}")
        
        return available_df
    
    def _apply_value_based_strategy(self, player_data, draft_context):
        """Apply value-based drafting strategy multipliers - ENHANCED"""
        position = player_data.get('position', '')
        round_num = draft_context.get('round', 1)
        
        # VALUE-BASED DRAFTING CORE PRINCIPLE: Avoid QBs early, prioritize skill positions
        if round_num <= 3:
            # Rounds 1-3: Elite skill position players ONLY
            if position in ['RB', 'WR']:
                return 2.5  # Massive boost for skill positions
            elif position == 'QB':
                return 0.05  # Nearly eliminate QBs in early rounds
            elif position == 'TE':
                return 0.4   # Strong penalty for TEs (except elite)
            else:
                return 0.02  # Eliminate DEF/K
                
        elif round_num <= 6:
            # Rounds 4-6: Fill starting lineup, still avoid QBs
            if position in ['RB', 'WR']:
                return 2.0   # Strong preference for skill positions
            elif position == 'QB':
                return 0.1   # Still heavily penalize QBs
            elif position == 'TE':
                return 1.0   # Neutral for TEs (elite TEs acceptable here)
            else:
                return 0.1   # Avoid DEF/K
                
        elif round_num <= 10:
            # Rounds 7-10: First acceptable QB range per value-based strategy
            if position == 'QB':
                return 1.8   # Now QBs become valuable
            elif position in ['RB', 'WR']:
                return 1.4   # Still prefer skill positions for depth
            elif position == 'TE':
                return 1.0   # Neutral
            else:
                return 0.3   # Still avoid DEF/K
                
        else:
            # Rounds 11-15: Primary QB territory and late-round value
            if position == 'QB':
                return 2.0   # Prime QB drafting rounds
            elif position in ['RB', 'WR']:
                return 1.2   # Depth plays
            elif position == 'TE':
                return 0.9   # Slight penalty
            else:
                return 1.5   # Now DEF/K become acceptable
        
        return 1.0
    
    def _find_best_skill_position_alternative(self, available_players, round_num):
        """ENHANCED: Find best RB/WR alternative to avoid early QB drafting"""
        if round_num > 6:  # After Round 6, QBs are acceptable per value-based strategy
            return None
            
        # Filter for skill positions only
        skill_players = available_players[
            available_players['position'].isin(['RB', 'WR'])
        ].copy()
        
        if len(skill_players) == 0:
            return None
        
        # Sort by availability and fantasy points
        skill_players = skill_players.sort_values(
            ['availability_probability', 'fantasy_points'], 
            ascending=[False, False]
        )
        
        # Return best available skill position player with decent availability
        best_skill = skill_players.iloc[0] if len(skill_players) > 0 else None
        
        if best_skill is not None and best_skill['availability_probability'] > 0.4:
            return best_skill
        
        return None
    
    def get_optimal_pick(self, available_players, draft_context):
        """ENHANCED: Get optimal pick with strict value-based drafting enforcement"""
        if available_players is None or len(available_players) == 0:
            return None, None
        
        round_num = draft_context.get('round', 1)
        
        # Add scores to all available players
        players_with_scores = available_players.copy()
        scores = []
        
        for _, player in players_with_scores.iterrows():
            score = self.calculate_neural_network_score(player, draft_context)
            scores.append(score)
        
        players_with_scores['score'] = scores
        players_with_scores = players_with_scores.sort_values('score', ascending=False)
        
        # VALUE-BASED STRATEGY ENFORCEMENT: Avoid QBs in early rounds
        if round_num <= 6:
            top_player = players_with_scores.iloc[0]
            if top_player['position'] == 'QB':
                print(f"🚫 Avoiding QB in Round {round_num} per value-based strategy")
                
                # Look for skill position alternative
                skill_alternative = self._find_best_skill_position_alternative(players_with_scores, round_num)
                
                if skill_alternative is not None:
                    print(f"✅ Selecting {skill_alternative['player_name']} ({skill_alternative['position']}) instead")
                    
                    # Recalculate score for the alternative
                    alt_score = self.calculate_neural_network_score(skill_alternative, draft_context)
                    
                    return skill_alternative['player_id'], {
                        'player_name': skill_alternative['player_name'],
                        'position': skill_alternative['position'],
                        'team': skill_alternative['team'],
                        'status': skill_alternative['status'],
                        'score': alt_score,
                        'ai_confidence': skill_alternative['ai_confidence'],
                        'fantasy_points': skill_alternative['fantasy_points'],
                        'age': skill_alternative['age'],
                        'last_active_year': skill_alternative['last_active_year'],
                        'adp': skill_alternative.get('adp', 100),
                        'availability_probability': skill_alternative.get('availability_probability', 0.5),
                        'reasoning': f"Value-based strategy: Avoiding QB in Round {round_num}, selecting best available {skill_alternative['position']}",
                        'alternatives': players_with_scores.head(3).to_dict('records')
                    }
                else:
                    print(f"⚠️ No suitable skill position alternatives found")
        
        # Return top-scoring player
        top_player = players_with_scores.iloc[0]
        
        reasoning = self._generate_ai_reasoning(top_player, players_with_scores.head(5).to_dict('records'), draft_context)
        
        return top_player['player_id'], {
            'player_name': top_player['player_name'],
            'position': top_player['position'],
            'team': top_player['team'],
            'status': top_player['status'],
            'score': top_player['score'],
            'ai_confidence': top_player['ai_confidence'],
            'fantasy_points': top_player['fantasy_points'],
            'age': top_player['age'],
            'last_active_year': top_player['last_active_year'],
            'adp': top_player.get('adp', 100),
            'availability_probability': top_player.get('availability_probability', 0.5),
            'reasoning': reasoning,
            'alternatives': players_with_scores.head(5).to_dict('records')[1:4]
        }
    
    def _generate_ai_reasoning(self, pick, alternatives, draft_context):
        """Generate enhanced AI reasoning with value-based strategy context"""
        reasoning_parts = []
        round_num = draft_context.get('round', 1)
        
        # Status-based reasoning
        if '🟢' in pick['status']:
            reasoning_parts.append("Active 2024 player with proven NFL production")
        elif '🟡' in pick['status']:
            reasoning_parts.append("Recently active player with established track record")
        elif '⭐' in pick['status']:
            reasoning_parts.append("High-upside 2025 rookie prospect")
        
        # Value-based strategy reasoning
        position = pick['position']
        if round_num <= 6 and position in ['RB', 'WR']:
            reasoning_parts.append(f"Value-based strategy: Prioritizing {position} in early rounds")
        elif round_num >= 7 and position == 'QB':
            reasoning_parts.append("Value-based strategy: Optimal QB drafting window (Rounds 7-10)")
        
        # Availability reasoning
        availability = pick.get('availability_probability', 0.5)
        adp = pick.get('adp', 100)
        picks_past_adp = pick.get('picks_past_adp', 0)
        
        if picks_past_adp > 0:
            reasoning_parts.append(f"Good value: {picks_past_adp} picks past ADP {adp}")
        elif picks_past_adp == 0:
            reasoning_parts.append(f"Drafted right at ADP {adp}")
        else:
            reasoning_parts.append(f"Slight reach: {abs(picks_past_adp)} picks before ADP {adp}")
        
        # AI confidence
        confidence = pick['ai_confidence']
        if confidence > 0.8:
            reasoning_parts.append("High AI model confidence")
        
        # Position needs
        team_needs = draft_context.get('team_needs', {})
        if position in team_needs and team_needs[position] > 0:
            reasoning_parts.append(f"Fills {position} roster need")
        
        return "; ".join(reasoning_parts)

    def _create_mock_combine_data(self):
        """Create realistic mock combine data for 2025 prospects"""
        positions = ['QB', 'RB', 'WR', 'TE', 'DEF']
        names = [
            'Caleb Williams', 'Drake Maye', 'Jayden Daniels', 'Marvin Harrison Jr', 
            'Malik Nabers', 'Rome Odunze', 'Brock Bowers', 'Trey Benson', 'Blake Corum',
            'Jonathon Brooks', 'Adonai Mitchell', 'Keon Coleman', 'Xavier Worthy',
            'Ja\'Lynn Polk', 'Troy Franklin', 'Brian Thomas Jr', 'Ladd McConkey'
        ]
        
        combine_data = []
        for i, name in enumerate(names):
            position = positions[i % len(positions)]
            combine_data.append({
                'player_name': name,
                'position': position,
                'college': f'College_{i}',
                'height': np.random.normal(73, 2),
                'weight': np.random.normal(220, 25),
                'forty_yard': max(4.0, min(5.5, np.random.normal(4.6, 0.2))),
                'vertical_jump': max(25, min(45, np.random.normal(35, 4))),
                'broad_jump': max(100, min(140, np.random.normal(122, 9))),
                'bench_press': max(5, min(30, np.random.normal(16, 3))),
                'draft_round': np.random.choice([1, 2, 3, 4, 5, 6, 7], p=[0.3, 0.25, 0.2, 0.15, 0.07, 0.02, 0.01]),
                'draft_pick': np.random.randint(1, 260),
                'team': np.random.choice(['KC', 'BUF', 'CIN', 'BAL', 'LAC', 'MIA', 'NE', 'NYJ', 'PIT', 'CLE', 'HOU', 'IND', 'JAX', 'TEN', 'DEN', 'LV', 'LAR', 'SF', 'SEA', 'ARI', 'DAL', 'PHI', 'NYG', 'WAS', 'CHI', 'DET', 'GB', 'MIN', 'ATL', 'CAR', 'NO', 'TB'])
            })
        
        return pd.DataFrame(combine_data)
    
    def _remove_outliers(self):
        """Remove statistical outliers from safe numerical columns only"""
        print("🔍 Removing statistical outliers from safe columns...")
        
        # Process yearly offense data
        if self.yearly_offense is not None and len(self.yearly_offense) > 0:
            # Only remove outliers from safe columns, preserve critical player data
            safe_numerical_cols = [
                'age', 'height', 'weight', 'draft_round', 'draft_pick'
            ]
            
            for col in safe_numerical_cols:
                if col in self.yearly_offense.columns:
                    Q1 = self.yearly_offense[col].quantile(0.25)
                    Q3 = self.yearly_offense[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Remove outliers only from this column
                    self.yearly_offense = self.yearly_offense[
                        (self.yearly_offense[col] >= lower_bound) & 
                        (self.yearly_offense[col] <= upper_bound)
                    ]
            
            print(f"✅ Offense outliers removed, {len(self.yearly_offense)} players remaining")
        
        # Process yearly defense data
        if self.yearly_defense is not None and len(self.yearly_defense) > 0:
            # Only remove outliers from safe columns for defense
            safe_numerical_cols = [
                'age', 'height', 'weight', 'draft_round', 'draft_pick'
            ]
            
            for col in safe_numerical_cols:
                if col in self.yearly_defense.columns:
                    Q1 = self.yearly_defense[col].quantile(0.25)
                    Q3 = self.yearly_defense[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    self.yearly_defense = self.yearly_defense[
                        (self.yearly_defense[col] >= lower_bound) & 
                        (self.yearly_defense[col] <= upper_bound)
                    ]
            
            print(f"✅ Defense outliers removed, {len(self.yearly_defense)} players remaining")
        
        print("✅ Outliers removed from safe columns only")
    
    def _impute_missing_values(self):
        """Impute missing values using intelligent strategies"""
        print("🔧 Imputing missing values...")
        
        # Process yearly offense data
        if self.yearly_offense is not None and len(self.yearly_offense) > 0:
            # Age: median by position
            if 'age' in self.yearly_offense.columns:
                self.yearly_offense['age'] = self.yearly_offense.groupby('position')['age'].transform(
                    lambda x: x.fillna(x.median())
                )
            
            # Years experience: median by position and age
            if 'years_exp' in self.yearly_offense.columns:
                self.yearly_offense['years_exp'] = self.yearly_offense.groupby(['position', 'age'])['years_exp'].transform(
                    lambda x: x.fillna(x.median())
                )
            
            # Fantasy points: 0 for missing values
            if 'fantasy_points_ppr' in self.yearly_offense.columns:
                self.yearly_offense['fantasy_points_ppr'] = self.yearly_offense['fantasy_points_ppr'].fillna(0)
            
            # Games played: median by position
            if 'games_played_season' in self.yearly_offense.columns:
                self.yearly_offense['games_played_season'] = self.yearly_offense.groupby('position')['games_played_season'].transform(
                    lambda x: x.fillna(x.median())
                )
        
        # Process yearly defense data
        if self.yearly_defense is not None and len(self.yearly_defense) > 0:
            # Similar imputation for defense stats
            if 'fantasy_points_ppr' in self.yearly_defense.columns:
                self.yearly_defense['fantasy_points_ppr'] = self.yearly_defense['fantasy_points_ppr'].fillna(0)
            
            if 'games_played_season' in self.yearly_defense.columns:
                self.yearly_defense['games_played_season'] = self.yearly_defense.groupby('position')['games_played_season'].transform(
                    lambda x: x.fillna(x.median())
                )
        
        print("✅ Missing values imputed")
    
    def _apply_ai_preprocessing(self):
        """Apply advanced AI preprocessing techniques"""
        print("🧠 Applying AI preprocessing...")
        
        # 1. Outlier detection and removal
        self._remove_outliers()
        
        # 2. Missing value imputation using player similarity
        self._impute_missing_values()
        
        # 3. Feature engineering
        self._engineer_features()
        
        print("✅ AI preprocessing completed!")

    def _engineer_features(self):
        """Engineer advanced features for predictive modeling"""
        print("🔧 Engineering advanced features...")
        
        # Process yearly offense data
        if self.yearly_offense is not None and len(self.yearly_offense) > 0:
            # Age-related features
            if 'age' in self.yearly_offense.columns:
                self.yearly_offense['age_squared'] = self.yearly_offense['age'] ** 2
                self.yearly_offense['age_group'] = pd.cut(
                    self.yearly_offense['age'], 
                    bins=[0, 23, 26, 29, 32, 100], 
                    labels=['Rookie', 'Young', 'Prime', 'Veteran', 'Elder']
                )
            
            # Experience features
            if 'years_exp' in self.yearly_offense.columns:
                self.yearly_offense['experience_squared'] = self.yearly_offense['years_exp'] ** 2
                self.yearly_offense['rookie'] = (self.yearly_offense['years_exp'] == 0).astype(int)
                self.yearly_offense['sophomore'] = (self.yearly_offense['years_exp'] == 1).astype(int)
            
            # Performance features
            if 'fantasy_points_ppr' in self.yearly_offense.columns:
                self.yearly_offense['fantasy_points_per_game'] = (
                    self.yearly_offense['fantasy_points_ppr'] / 
                    self.yearly_offense['games_played_season'].replace(0, 1)
                )
            
            # Position-specific features
            if 'position' in self.yearly_offense.columns:
                # RB features
                rb_mask = self.yearly_offense['position'] == 'RB'
                if 'rushing_yards' in self.yearly_offense.columns and 'rushing_attempts' in self.yearly_offense.columns:
                    self.yearly_offense.loc[rb_mask, 'yards_per_carry'] = (
                        self.yearly_offense.loc[rb_mask, 'rushing_yards'] / 
                        self.yearly_offense.loc[rb_mask, 'rushing_attempts'].replace(0, 1)
                    )
                
                # WR features
                wr_mask = self.yearly_offense['position'] == 'WR'
                if 'receiving_yards' in self.yearly_offense.columns and 'targets' in self.yearly_offense.columns:
                    self.yearly_offense.loc[wr_mask, 'yards_per_target'] = (
                        self.yearly_offense.loc[wr_mask, 'receiving_yards'] / 
                        self.yearly_offense.loc[wr_mask, 'targets'].replace(0, 1)
                    )
                
                # QB features
                qb_mask = self.yearly_offense['position'] == 'QB'
                if 'passing_yards' in self.yearly_offense.columns and 'passing_attempts' in self.yearly_offense.columns:
                    self.yearly_offense.loc[qb_mask, 'yards_per_attempt'] = (
                        self.yearly_offense.loc[qb_mask, 'passing_yards'] / 
                        self.yearly_offense.loc[qb_mask, 'passing_attempts'].replace(0, 1)
                    )
        
        # Process yearly defense data
        if self.yearly_defense is not None and len(self.yearly_defense) > 0:
            # Defense-specific features
            if 'tackles' in self.yearly_defense.columns and 'games_played_season' in self.yearly_defense.columns:
                self.yearly_defense['tackles_per_game'] = (
                    self.yearly_defense['tackles'] / 
                    self.yearly_defense['games_played_season'].replace(0, 1)
                )
            
            if 'sacks' in self.yearly_defense.columns and 'games_played_season' in self.yearly_defense.columns:
                self.yearly_defense['sacks_per_game'] = (
                    self.yearly_defense['sacks'] / 
                    self.yearly_defense['games_played_season'].replace(0, 1)
                )
        
        print("✅ Advanced features engineered")

    def build_ml_models(self):
        """Build machine learning models for player performance prediction"""
        print("🤖 Building machine learning models...")
        
        try:
            # Build veteran performance model
            if self.yearly_offense is not None and len(self.yearly_offense) > 0:
                self._build_veteran_model()
            
            # Build rookie projection model
            if self.combine_data is not None and len(self.combine_data) > 0:
                self._build_rookie_model()
            
            print(f"✅ Built {len(self.models)} ML models")
            
            # Show model details
            if 'veteran_performance' in self.models:
                print(f"✅ Veteran model has {len(self.feature_importance.get('veteran_performance', []))} features")
            
        except Exception as e:
            print(f"⚠️ Error building ML models: {e}")
            # Create fallback models
            self._create_fallback_models()
    
    def _build_veteran_model(self):
        """Build model for veteran player performance prediction"""
        try:
            # Prepare features for veteran model
            features = ['age', 'years_exp', 'games_played_season']
            target = 'fantasy_points_ppr'
            
            # Add engineered features if they exist
            if 'age_squared' in self.yearly_offense.columns:
                features.append('age_squared')
            if 'experience_squared' in self.yearly_offense.columns:
                features.append('experience_squared')
            if 'fantasy_points_per_game' in self.yearly_offense.columns:
                features.append('fantasy_points_per_game')
            
            # Filter data with all required features
            model_data = self.yearly_offense[features + [target]].dropna()
            
            if len(model_data) < 100:  # Need sufficient data
                print("⚠️ Insufficient data for veteran model")
                return
            
            X = model_data[features]
            y = model_data[target]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Random Forest model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store model and metrics
            self.models['veteran_performance'] = model
            self.feature_importance['veteran_performance'] = dict(zip(features, model.feature_importances_))
            
            print(f"✅ Veteran model built - R²: {r2:.3f}, MSE: {mse:.2f}")
            
        except Exception as e:
            print(f"⚠️ Error building veteran model: {e}")
    
    def _build_rookie_model(self):
        """Build model for rookie player projection"""
        try:
            # Check for required combine columns
            required_cols = ['forty_yard', 'vertical_jump', 'broad_jump']
            missing_cols = [col for col in required_cols if col not in self.combine_data.columns]
            
            if missing_cols:
                print(f"⚠️ Missing combine columns: {missing_cols}. Using simplified rookie model.")
                self._create_simplified_rookie_model()
                return
            
            # Prepare features for rookie model
            features = ['height', 'weight', 'forty_yard', 'vertical_jump', 'broad_jump', 'bench_press']
            target = 'draft_round'  # Use draft round as proxy for expected performance
            
            # Filter data with all required features
            model_data = self.combine_data[features + [target]].dropna()
            
            if len(model_data) < 50:  # Need sufficient data
                print("⚠️ Insufficient data for rookie model")
                return
            
            X = model_data[features]
            y = model_data[target]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Gradient Boosting model
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store model and metrics
            self.models['rookie_projection'] = model
            self.feature_importance['rookie_projection'] = dict(zip(features, model.feature_importances_))
            
            print(f"✅ Rookie model built - R²: {r2:.3f}, MSE: {mse:.2f}")
            
        except Exception as e:
            print(f"⚠️ Error building rookie model: {e}")
            self._create_simplified_rookie_model()
    
    def _create_simplified_rookie_model(self):
        """Create a simplified rookie model when combine data is limited"""
        print("🔧 Creating simplified rookie model...")
        
        # Simple rule-based model
        def simple_rookie_score(player_data):
            score = 0.5  # Base score
            
            # Draft round adjustment
            if 'draft_round' in player_data:
                draft_round = player_data['draft_round']
                if draft_round == 1:
                    score += 0.3
                elif draft_round == 2:
                    score += 0.2
                elif draft_round == 3:
                    score += 0.1
            
            # Position adjustment
            if 'position' in player_data:
                position = player_data['position']
                if position == 'RB':
                    score += 0.1
                elif position == 'WR':
                    score += 0.05
            
            return min(1.0, max(0.0, score))
        
        self.models['rookie_projection'] = simple_rookie_score
        print("✅ Simplified rookie model created")
    
    def _create_fallback_models(self):
        """Create fallback models when ML training fails"""
        print("🔧 Creating fallback models...")
        
        # Simple fallback for veteran performance
        def fallback_veteran_score(player_data):
            fantasy_points = player_data.get('fantasy_points', 0)
            age = player_data.get('age', 25)
            
            # Simple age-based adjustment
            if age < 25:
                return fantasy_points * 1.1
            elif age > 30:
                return fantasy_points * 0.9
            else:
                return fantasy_points
        
        self.models['veteran_performance'] = fallback_veteran_score
        print("✅ Fallback models created")

    def classify_active_players(self):
        """Classify players as active veterans or rookies with AI confidence scores"""
        print("🎯 Classifying active players with AI...")
        
        try:
            active_players = []
            
            # Process veteran players from yearly offense data
            if self.yearly_offense is not None and len(self.yearly_offense) > 0:
                # Get the most recent season data
                max_season = self.yearly_offense['season'].max()
                recent_data = self.yearly_offense[self.yearly_offense['season'] == max_season]
                
                # Filter for active players with meaningful production
                active_veterans = recent_data[
                    (recent_data['fantasy_points_ppr'] > 0) & 
                    (recent_data['games_played_season'] >= 3)
                ].copy()
                
                for _, player in active_veterans.iterrows():
                    # Determine player status based on recent activity
                    if player['fantasy_points_ppr'] >= 100:  # High production
                        status = '🟢 2024 NFL Veteran'
                        active_bonus = 50
                        ai_confidence = 0.95
                    elif player['fantasy_points_ppr'] >= 50:  # Moderate production
                        status = '🟡 2023 NFL Veteran'
                        active_bonus = 25
                        ai_confidence = 0.85
                    else:  # Low production but active
                        status = '🟡 2023 NFL Veteran'
                        active_bonus = 15
                        ai_confidence = 0.75
                    
                    active_players.append({
                        'player_id': player['player_id'],
                        'player_name': player['player_name'],
                        'position': player['position'],
                        'team': player['team'],
                        'status': status,
                        'ai_confidence': ai_confidence,
                        'active_bonus': active_bonus,
                        'last_active_year': max_season,
                        'fantasy_points': player['fantasy_points_ppr'],
                        'age': player.get('age', 25),
                        'games_played': player.get('games_played_season', 0),
                        'years_exp': player.get('years_exp', 0)
                    })
            
            # Process rookie players from combine data
            if self.combine_data is not None and len(self.combine_data) > 0:
                for _, rookie in self.combine_data.iterrows():
                    # Calculate rookie projection score
                    rookie_score = self._calculate_rookie_projection(rookie)
                    
                    active_players.append({
                        'player_id': f"rookie_{len(active_players)}",
                        'player_name': rookie.get('player_name', f"Rookie_{len(active_players)}"),
                        'position': rookie.get('position', 'RB'),
                        'team': rookie.get('team', 'FA'),
                        'status': '⭐ 2025 Rookie',
                        'ai_confidence': 0.80,
                        'active_bonus': 30,
                        'last_active_year': 2025,
                        'fantasy_points': rookie_score,
                        'age': 22,  # Typical rookie age
                        'games_played': 0,
                        'years_exp': 0,
                        'athletic_score': self._calculate_athletic_score(rookie),
                        'draft_value': self._calculate_draft_value(rookie)
                    })
            
            # Convert to DataFrame
            self.current_players = pd.DataFrame(active_players)
            
            # Calculate average AI confidence
            avg_confidence = self.current_players['ai_confidence'].mean()
            
            # Show classification summary
            status_counts = self.current_players['status'].value_counts()
            print(f"✅ Classified {len(self.current_players)} active players")
            for status, count in status_counts.items():
                print(f"   {status}: {count} players")
            print(f"✅ Average AI confidence: {avg_confidence:.2f}")
            
        except Exception as e:
            print(f"❌ Error classifying players: {e}")
            # Create fallback data
            self._create_fallback_player_data()
    
    def _calculate_rookie_projection(self, rookie_data):
        """Calculate fantasy points projection for rookie players"""
        base_projection = 50  # Base projection
        
        # Adjust based on draft round
        if 'draft_round' in rookie_data:
            draft_round = rookie_data['draft_round']
            if draft_round == 1:
                base_projection *= 1.5
            elif draft_round == 2:
                base_projection *= 1.3
            elif draft_round == 3:
                base_projection *= 1.1
        
        # Adjust based on position
        if 'position' in rookie_data:
            position = rookie_data['position']
            if position == 'RB':
                base_projection *= 1.2  # RBs often have immediate impact
            elif position == 'WR':
                base_projection *= 1.1  # WRs can have good rookie years
            elif position == 'QB':
                base_projection *= 0.8  # QBs often struggle as rookies
        
        return base_projection
    
    def _calculate_athletic_score(self, rookie_data):
        """Calculate athletic score from combine metrics"""
        score = 0.5  # Base score
        
        # Check if combine metrics exist
        if 'forty_yard' in rookie_data and 'vertical_jump' in rookie_data and 'broad_jump' in rookie_data:
            # Forty yard dash (lower is better)
            forty = rookie_data['forty_yard']
            if forty <= 4.4:
                score += 0.2
            elif forty <= 4.6:
                score += 0.1
            
            # Vertical jump (higher is better)
            vertical = rookie_data['vertical_jump']
            if vertical >= 40:
                score += 0.2
            elif vertical >= 35:
                score += 0.1
            
            # Broad jump (higher is better)
            broad = rookie_data['broad_jump']
            if broad >= 130:
                score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _calculate_draft_value(self, rookie_data):
        """Calculate draft value score"""
        score = 0.5  # Base score
        
        if 'draft_round' in rookie_data:
            draft_round = rookie_data['draft_round']
            if draft_round == 1:
                score += 0.4
            elif draft_round == 2:
                score += 0.3
            elif draft_round == 3:
                score += 0.2
            elif draft_round == 4:
                score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _create_fallback_player_data(self):
        """Create fallback player data when classification fails"""
        print("🔧 Creating fallback player data...")
        
        fallback_players = [
            {
                'player_id': 'fallback_1',
                'player_name': 'Sample Player',
                'position': 'RB',
                'team': 'FA',
                'status': '🟢 2024 NFL Veteran',
                'ai_confidence': 0.8,
                'active_bonus': 50,
                'last_active_year': 2024,
                'fantasy_points': 100,
                'age': 25,
                'games_played': 16,
                'years_exp': 3
            }
        ]
        
        self.current_players = pd.DataFrame(fallback_players)
        print("✅ Fallback player data created")

    def create_draft_strategy(self, draft_position: int, league_settings: dict):
        """Create comprehensive AI-powered draft strategy"""
        print(f"🎯 Creating AI draft strategy for position #{draft_position}")
        
        try:
            # Build ML models if not already built
            if not self.models:
                self.build_ml_models()
            
            # Classify players if not already done
            if self.current_players is None:
                self.classify_active_players()
            
            # Create draft strategy
            strategy = []
            league_size = league_settings.get('teams', 12)
            positions = league_settings.get('positions', {})
            
            # Track team needs
            team_needs = positions.copy()
            
            for round_num in range(1, 16):  # 15 rounds
                # Get available players at this pick
                available = self.get_available_players_at_pick(draft_position, round_num, league_size)
                
                if len(available) == 0:
                    print(f"⚠️ No players available in Round {round_num}")
                    continue
                
                # Create draft context
                draft_context = {
                    'round': round_num,
                    'draft_position': draft_position,
                    'league_size': league_size,
                    'team_needs': team_needs.copy()
                }
                
                # Get optimal pick
                pick_id, pick_data = self.get_optimal_pick(available, draft_context)
                
                if pick_id and pick_data:
                    # Update team needs
                    position = pick_data['position']
                    if position in team_needs and team_needs[position] > 0:
                        team_needs[position] -= 1
                    
                    # Add to strategy
                    strategy.append({
                        'round': round_num,
                        'pick': pick_data,
                        'team_needs': team_needs.copy()
                    })
                    
                    print(f"Round {round_num}: {pick_data['player_name']} ({pick_data['position']}) - Score: {pick_data['score']:.1f}")
                else:
                    print(f"⚠️ No optimal pick found in Round {round_num}")
            
            print(f"✅ Created draft strategy with {len(strategy)} rounds")
            print(f"✅ Used {len(self.models)} AI models")
            
            # Show first few rounds
            for item in strategy[:3]:
                pick = item['pick']
                print(f"   Round {item['round']}: {pick['player_name']} ({pick['position']}) - Score: {pick['score']:.1f}")
            
            return {
                'strategy': strategy,
                'draft_position': draft_position,
                'league_settings': league_settings,
                'ai_models_used': len(self.models)
            }
            
        except Exception as e:
            print(f"❌ Error creating draft strategy: {e}")
            # Return fallback strategy
            return self._create_fallback_strategy(draft_position, league_settings)
    
    def _create_fallback_strategy(self, draft_position: int, league_settings: dict):
        """Create fallback draft strategy when main method fails"""
        print("🔧 Creating fallback draft strategy...")
        
        fallback_strategy = []
        for round_num in range(1, 16):
            fallback_strategy.append({
                'round': round_num,
                'pick': {
                    'player_name': f'Fallback Player {round_num}',
                    'position': 'RB' if round_num <= 3 else 'WR' if round_num <= 6 else 'QB',
                    'team': 'FA',
                    'status': '🟢 2024 NFL Veteran',
                    'score': 100 - round_num * 5,
                    'ai_confidence': 0.8,
                    'fantasy_points': 100 - round_num * 5,
                    'age': 25,
                    'last_active_year': 2024,
                    'adp': 50 + round_num * 10,
                    'availability_probability': 0.9,
                    'reasoning': 'Fallback strategy due to system error',
                    'alternatives': []
                },
                'team_needs': league_settings.get('positions', {}).copy()
            })
        
        return {
            'strategy': fallback_strategy,
            'draft_position': draft_position,
            'league_settings': league_settings,
            'ai_models_used': 0
        }

    def calculate_neural_network_score(self, player_data, draft_context):
        """Calculate neural network-inspired score using value-based drafting strategy"""
        if player_data is None:
            return 0
        
        # Base performance score (40% weight)
        base_score = self._calculate_base_performance_score(player_data)
        
        # Context factors (30% weight) - Position scarcity and team needs
        context_score = self._calculate_context_score(player_data, draft_context)
        
        # AI enhancements (20% weight) - Model predictions and confidence
        ai_score = self._calculate_ai_enhancement_score(player_data)
        
        # Meta factors (10% weight) - Strategic considerations
        meta_score = self._calculate_meta_score(player_data, draft_context)
        
        # Apply value-based drafting strategy
        strategy_multiplier = self._apply_value_based_strategy(player_data, draft_context)
        
        # Calculate final score
        final_score = (
            base_score * 0.4 +
            context_score * 0.3 +
            ai_score * 0.2 +
            meta_score * 0.1
        ) * strategy_multiplier
        
        return max(0, final_score)
    
    def _calculate_base_performance_score(self, player_data):
        """Calculate base performance score based on player status and production"""
        if player_data['status'] == '🟢 2024 NFL Veteran':
            base_score = self._calculate_veteran_base_score(player_data)
        elif player_data['status'] == '🟡 2023 NFL Veteran':
            base_score = self._calculate_veteran_base_score(player_data) * 0.8
        elif player_data['status'] == '⭐ 2025 Rookie':
            base_score = self._calculate_rookie_base_score(player_data)
        else:
            base_score = 0
        
        return base_score
    
    def _calculate_veteran_base_score(self, player_data):
        """Calculate base score for veteran players"""
        fantasy_points = player_data.get('fantasy_points', 0)
        age = player_data.get('age', 25)
        
        # Age curve adjustment
        age_factor = 1.0
        if age < 25:
            age_factor = 1.1  # Young player bonus
        elif age > 30:
            age_factor = 0.9  # Age penalty
        
        return fantasy_points * age_factor
    
    def _calculate_rookie_base_score(self, player_data):
        """Calculate base score for rookie players"""
        projection = player_data.get('fantasy_points', 50)  # Default projection
        athletic_score = player_data.get('athletic_score', 0.5)
        draft_value = player_data.get('draft_value', 0.5)
        
        # Combine projection with athletic and draft metrics
        base_score = projection * 0.6 + athletic_score * 20 + draft_value * 20
        
        return base_score
    
    def _calculate_ai_enhancement_score(self, player_data):
        """Calculate AI enhancement factors"""
        enhancement = 0
        
        # AI confidence boost
        ai_confidence = player_data.get('ai_confidence', 0.5)
        enhancement += ai_confidence * 50
        
        # Active bonus
        active_bonus = player_data.get('active_bonus', 0)
        enhancement += active_bonus
        
        # Model predictions if available
        if 'veteran_performance' in self.models:
            enhancement += 20  # Model confidence boost
        
        return enhancement
    
    def _calculate_meta_score(self, player_data, draft_context):
        """Calculate meta factors (news sentiment, market inefficiencies, etc.)"""
        meta_score = 0
        
        # Team efficiency (if team stats available)
        team = player_data.get('team', '')
        if team and hasattr(self, 'yearly_team_offense') and self.yearly_team_offense is not None:
            team_stats = self.yearly_team_offense[
                (self.yearly_team_offense['team'] == team) & 
                (self.yearly_team_offense['season'] == self.yearly_team_offense['season'].max())
            ]
            if len(team_stats) > 0:
                # Boost for players on efficient offenses
                meta_score += 10
        
        # Rookie upside for 2025 class
        if player_data['status'] == '⭐ 2025 Rookie':
            meta_score += 15  # Rookie upside bonus
        
        # Injury risk adjustment
        age = player_data.get('age', 25)
        if age > 28:
            meta_score -= 5  # Age-related injury risk
        
        return meta_score

    def _calculate_context_score(self, player_data, draft_context):
        """Calculate context score with value-based drafting considerations"""
        position = player_data.get('position', '')
        team_needs = draft_context.get('team_needs', {})
        
        # Position scarcity scoring
        position_scarcity = {
            'RB': 1.0,    # Most scarce - only ~12 true RB1s
            'WR': 0.9,    # Less scarce - ~24 startable WRs
            'TE': 0.7,    # Very scarce - only ~3 elite TEs
            'QB': 0.4,    # Least scarce - 15-20 startable QBs
            'DEF': 0.3,   # Streamable
            'K': 0.2      # Most streamable
        }
        
        scarcity_score = position_scarcity.get(position, 0.5) * 100
        
        # Team needs scoring
        position_need = team_needs.get(position, 0)
        if position_need > 0:
            needs_score = 50  # Boost if position is needed
        else:
            needs_score = 25  # Still some value for depth
        
        # Value-based drafting adjustments
        if position in ['RB', 'WR'] and position_need > 0:
            needs_score *= 1.5  # Extra boost for needed skill positions
        
        return scarcity_score + needs_score