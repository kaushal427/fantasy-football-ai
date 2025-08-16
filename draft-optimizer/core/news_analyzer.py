import requests
import json
from typing import Dict, List, Optional
from textblob import TextBlob
import time
import os
from datetime import datetime, timedelta

class NewsAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('NEWS_API_KEY')
        self.base_url = "https://newsapi.org/v2/everything"
        self.cache = {}
        self.cache_duration = 3600  # 1 hour cache
        
    def get_player_news(self, player_name: str, days_back: int = 7) -> List[Dict]:
        """Get recent news about a specific player"""
        cache_key = f"{player_name}_{days_back}"
        
        # Check cache first
        if cache_key in self.cache:
            cache_time, cache_data = self.cache[cache_key]
            if time.time() - cache_time < self.cache_duration:
                return cache_data
        
        if not self.api_key:
            # Return mock data if no API key
            return self._get_mock_news(player_name)
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Search for NFL news about the player
            query = f'"{player_name}" AND (NFL OR football)'
            
            params = {
                'q': query,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'relevancy',
                'apiKey': self.api_key,
                'pageSize': 10
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            # Process articles
            processed_articles = []
            for article in articles:
                processed_articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'source': article.get('source', {}).get('name', '')
                })
            
            # Cache the results
            self.cache[cache_key] = (time.time(), processed_articles)
            
            return processed_articles
            
        except Exception as e:
            print(f"Error fetching news for {player_name}: {e}")
            return self._get_mock_news(player_name)
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text using TextBlob"""
        blob = TextBlob(text)
        
        # Get polarity (-1 to 1) and subjectivity (0 to 1)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Convert to more intuitive scores
        sentiment_score = (polarity + 1) / 2  # Convert to 0-1 scale
        confidence = 1 - subjectivity  # Higher subjectivity = lower confidence
        
        return {
            'sentiment_score': sentiment_score,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'confidence': confidence
        }
    
    def get_player_sentiment(self, player_name: str) -> Dict[str, float]:
        """Get overall sentiment for a player based on recent news"""
        articles = self.get_player_news(player_name)
        
        if not articles:
            return {
                'sentiment_score': 0.5,  # Neutral
                'confidence': 0.0,
                'article_count': 0
            }
        
        # Analyze sentiment for each article
        sentiments = []
        for article in articles:
            # Combine title and description for analysis
            text = f"{article['title']} {article['description']}"
            sentiment = self.analyze_sentiment(text)
            sentiments.append(sentiment)
        
        # Calculate weighted average based on confidence
        if sentiments:
            total_weight = sum(s['confidence'] for s in sentiments)
            if total_weight > 0:
                weighted_sentiment = sum(
                    s['sentiment_score'] * s['confidence'] for s in sentiments
                ) / total_weight
                avg_confidence = sum(s['confidence'] for s in sentiments) / len(sentiments)
            else:
                weighted_sentiment = 0.5
                avg_confidence = 0.0
        else:
            weighted_sentiment = 0.5
            avg_confidence = 0.0
        
        return {
            'sentiment_score': weighted_sentiment,
            'confidence': avg_confidence,
            'article_count': len(articles)
        }
    
    def get_injury_news(self, player_name: str) -> List[Dict]:
        """Get injury-related news for a player"""
        articles = self.get_player_news(player_name)
        
        injury_keywords = [
            'injury', 'injured', 'hurt', 'out', 'questionable', 'doubtful',
            'concussion', 'hamstring', 'ankle', 'knee', 'shoulder', 'back',
            'IR', 'reserve', 'surgery', 'recovery', 'rehab'
        ]
        
        injury_articles = []
        for article in articles:
            text = f"{article['title']} {article['description']}".lower()
            if any(keyword in text for keyword in injury_keywords):
                injury_articles.append(article)
        
        return injury_articles
    
    def get_team_news(self, team_name: str, days_back: int = 7) -> List[Dict]:
        """Get recent news about a team"""
        cache_key = f"team_{team_name}_{days_back}"
        
        if cache_key in self.cache:
            cache_time, cache_data = self.cache[cache_key]
            if time.time() - cache_time < self.cache_duration:
                return cache_data
        
        if not self.api_key:
            return self._get_mock_team_news(team_name)
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            query = f'"{team_name}" AND NFL'
            
            params = {
                'q': query,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'relevancy',
                'apiKey': self.api_key,
                'pageSize': 15
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            processed_articles = []
            for article in articles:
                processed_articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'source': article.get('source', {}).get('name', '')
                })
            
            self.cache[cache_key] = (time.time(), processed_articles)
            return processed_articles
            
        except Exception as e:
            print(f"Error fetching team news for {team_name}: {e}")
            return self._get_mock_team_news(team_name)
    
    def _get_mock_news(self, player_name: str) -> List[Dict]:
        """Return mock news data for testing"""
        return [
            {
                'title': f'{player_name} shows great potential in training camp',
                'description': f'Recent reports indicate {player_name} is performing well in practice.',
                'content': f'{player_name} has been impressing coaches with their work ethic and skill.',
                'url': 'https://example.com/mock-news',
                'published_at': datetime.now().isoformat(),
                'source': 'Mock News'
            }
        ]
    
    def _get_mock_team_news(self, team_name: str) -> List[Dict]:
        """Return mock team news data for testing"""
        return [
            {
                'title': f'{team_name} preparing for upcoming season',
                'description': f'The {team_name} are working hard to improve their roster.',
                'content': f'Team management is optimistic about the upcoming season.',
                'url': 'https://example.com/mock-team-news',
                'published_at': datetime.now().isoformat(),
                'source': 'Mock News'
            }
        ]
