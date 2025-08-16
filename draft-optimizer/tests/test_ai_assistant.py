#!/usr/bin/env python3
"""
Test suite for the AI Draft Assistant
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ai_draft_assistant import AIDraftAssistant
import time

def test_ai_data_loading():
    """Test AI data loading and preprocessing"""
    print("🤖 Testing AI data loading and preprocessing...")
    
    try:
        ai_assistant = AIDraftAssistant()
        ai_assistant.load_all_data()
        
        # Check if all data files are loaded
        data_files = [
            ai_assistant.weekly_offense,
            ai_assistant.yearly_offense,
            ai_assistant.weekly_defense,
            ai_assistant.yearly_defense,
            ai_assistant.weekly_team_offense,
            ai_assistant.weekly_team_defense,
            ai_assistant.yearly_team_offense,
            ai_assistant.yearly_team_defense,
            ai_assistant.combine_data
        ]
        
        loaded_files = sum(1 for df in data_files if df is not None and len(df) > 0)
        print(f"✅ Loaded {loaded_files}/9 data files successfully")
        
        return True
    except Exception as e:
        print(f"✗ AI data loading failed: {e}")
        return False

def test_ml_models():
    """Test machine learning model building"""
    print("\n🧠 Testing machine learning model building...")
    
    try:
        ai_assistant = AIDraftAssistant()
        ai_assistant.load_all_data()
        ai_assistant.build_ml_models()
        
        # Check models
        models_built = len(ai_assistant.models)
        print(f"✅ Built {models_built} ML models")
        
        # Check feature importance
        if 'veteran' in ai_assistant.feature_importance:
            features = len(ai_assistant.feature_importance['veteran'])
            print(f"✅ Veteran model has {features} features")
        
        return True
    except Exception as e:
        print(f"✗ ML model building failed: {e}")
        return False

def test_player_classification():
    """Test AI player classification"""
    print("\n🎯 Testing AI player classification...")
    
    try:
        ai_assistant = AIDraftAssistant()
        ai_assistant.load_all_data()
        ai_assistant.build_ml_models()
        active_players = ai_assistant.classify_active_players()
        
        # Check player classification
        status_counts = active_players['status'].value_counts()
        print(f"✅ Classified {len(active_players)} active players")
        
        for status, count in status_counts.items():
            print(f"   {status}: {count} players")
        
        # Check AI confidence distribution
        avg_confidence = active_players['ai_confidence'].mean()
        print(f"✅ Average AI confidence: {avg_confidence:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Player classification failed: {e}")
        return False

def test_neural_network_scoring():
    """Test neural network-inspired scoring system"""
    print("\n🧠 Testing neural network scoring system...")
    
    try:
        ai_assistant = AIDraftAssistant()
        ai_assistant.load_all_data()
        ai_assistant.build_ml_models()
        active_players = ai_assistant.classify_active_players()
        
        # Test scoring for different player types
        draft_context = {
            'team_needs': {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'DEF': 1}
        }
        
        scores = []
        for _, player in active_players.head(10).iterrows():
            score = ai_assistant.calculate_neural_network_score(player, draft_context)
            scores.append(score)
        
        avg_score = np.mean(scores)
        print(f"✅ Average AI score: {avg_score:.1f}")
        print(f"✅ Score range: {min(scores):.1f} - {max(scores):.1f}")
        
        return True
    except Exception as e:
        print(f"✗ Neural network scoring failed: {e}")
        return False

def test_optimal_pick_selection():
    """Test optimal pick selection"""
    print("\n🎯 Testing optimal pick selection...")
    
    try:
        ai_assistant = AIDraftAssistant()
        ai_assistant.load_all_data()
        ai_assistant.build_ml_models()
        active_players = ai_assistant.classify_active_players()
        
        # Test optimal pick
        draft_context = {
            'draft_position': 1,
            'round': 1,
            'team_needs': {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'DEF': 1}
        }
        
        pick_id, pick_data = ai_assistant.get_optimal_pick(active_players, draft_context)
        
        if pick_id and pick_data:
            print(f"✅ Optimal pick: {pick_data['player_name']} ({pick_data['position']})")
            print(f"   AI Score: {pick_data['score']:.1f}")
            print(f"   AI Confidence: {pick_data['ai_confidence']:.1%}")
            print(f"   Status: {pick_data['status']}")
            print(f"   Reasoning: {pick_data['reasoning']}")
        else:
            print("✗ No optimal pick found")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Optimal pick selection failed: {e}")
        return False

def test_draft_strategy():
    """Test AI draft strategy creation"""
    print("\n🎯 Testing AI draft strategy creation...")
    
    try:
        ai_assistant = AIDraftAssistant()
        ai_assistant.load_all_data()
        ai_assistant.build_ml_models()
        ai_assistant.classify_active_players()
        
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
        
        strategy = ai_assistant.create_draft_strategy(1, league_settings)
        
        print(f"✅ Created draft strategy with {len(strategy['strategy'])} rounds")
        print(f"✅ Used {len(strategy['ai_models_used'])} AI models")
        
        # Show first few picks
        for i, strategy_item in enumerate(strategy['strategy'][:3]):
            pick = strategy_item['pick']
            print(f"   Round {strategy_item['round']}: {pick['player_name']} ({pick['position']}) - Score: {pick['score']:.1f}")
        
        return True
    except Exception as e:
        print(f"✗ Draft strategy creation failed: {e}")
        return False

def test_rookie_integration():
    """Test 2025 rookie integration"""
    print("\n⭐ Testing 2025 rookie integration...")
    
    try:
        ai_assistant = AIDraftAssistant()
        ai_assistant.load_all_data()
        ai_assistant.build_ml_models()
        active_players = ai_assistant.classify_active_players()
        
        # Check for rookies
        rookies = active_players[active_players['status'] == '⭐ 2025 Rookie']
        
        if len(rookies) > 0:
            print(f"✅ Found {len(rookies)} 2025 rookies")
            
            # Show top rookies
            top_rookies = rookies.sort_values('fantasy_points', ascending=False).head(3)
            for _, rookie in top_rookies.iterrows():
                print(f"   {rookie['player_name']} ({rookie['position']}) - Projection: {rookie['fantasy_points']:.1f}")
        else:
            print("⚠️ No rookies found")
        
        return True
    except Exception as e:
        print(f"✗ Rookie integration failed: {e}")
        return False

def test_ai_confidence_calculation():
    """Test AI confidence calculation"""
    print("\n🤖 Testing AI confidence calculation...")
    
    try:
        ai_assistant = AIDraftAssistant()
        ai_assistant.load_all_data()
        ai_assistant.build_ml_models()
        active_players = ai_assistant.classify_active_players()
        
        # Test confidence for different player types
        veteran_confidence = active_players[
            active_players['status'].str.contains('Veteran')
        ]['ai_confidence'].mean()
        
        rookie_confidence = active_players[
            active_players['status'].str.contains('Rookie')
        ]['ai_confidence'].mean()
        
        print(f"✅ Average veteran confidence: {veteran_confidence:.2f}")
        print(f"✅ Average rookie confidence: {rookie_confidence:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ AI confidence calculation failed: {e}")
        return False

def main():
    """Run all AI assistant tests"""
    print("🤖 AI-Powered Fantasy Football Draft Assistant - System Test")
    print("=" * 70)
    
    tests = [
        test_ai_data_loading,
        test_ml_models,
        test_player_classification,
        test_neural_network_scoring,
        test_optimal_pick_selection,
        test_draft_strategy,
        test_rookie_integration,
        test_ai_confidence_calculation
    ]
    
    passed = 0
    total = len(tests)
    
    start_time = time.time()
    
    for test in tests:
        if test():
            passed += 1
    
    end_time = time.time()
    test_duration = end_time - start_time
    
    print("\n" + "=" * 70)
    print(f"🤖 AI Test Results: {passed}/{total} tests passed")
    print(f"⏱️ Test duration: {test_duration:.1f} seconds")
    
    if passed == total:
        print("🎉 All AI tests passed! The AI assistant is ready to use.")
        print("\nTo run the AI-powered application:")
        print("streamlit run ai_app.py")
    else:
        print("⚠️ Some AI tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()
