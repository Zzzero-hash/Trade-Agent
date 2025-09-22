"""
Alternative Data Integration Features

This module implements alternative data integration including:
- Sentiment analysis features
- News embeddings and NLP features
- Macroeconomic indicators
- Social media sentiment
- Earnings and fundamental data integration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
import re
from collections import Counter

warnings.filterwarnings("ignore")


@dataclass
class AlternativeDataConfig:
    """Configuration for alternative data features."""

    sentiment_windows: List[int] = None
    news_embedding_dim: int = 128
    macro_indicators: List[str] = None
    social_platforms: List[str] = None
    fundamental_metrics: List[str] = None

    def __post_init__(self):
        if self.sentiment_windows is None:
            self.sentiment_windows = [1, 3, 7, 14, 30]
        if self.macro_indicators is None:
            self.macro_indicators = [
                "vix",
                "dxy",
                "yield_10y",
                "oil_price",
                "gold_price",
            ]
        if self.social_platforms is None:
            self.social_platforms = ["twitter", "reddit", "stocktwits"]
        if self.fundamental_metrics is None:
            self.fundamental_metrics = [
                "pe_ratio",
                "pb_ratio",
                "debt_equity",
                "roe",
                "roa",
            ]


class SentimentFeatures:
    """Sentiment analysis and text-based features."""

    def __init__(self, config: AlternativeDataConfig):
        self.config = config
        self._initialize_sentiment_lexicons()

    def _initialize_sentiment_lexicons(self):
        """Initialize sentiment lexicons and word lists."""
        # Financial sentiment words (simplified - in practice, use comprehensive lexicons)
        self.positive_words = {
            "bullish",
            "buy",
            "strong",
            "growth",
            "profit",
            "gain",
            "rise",
            "up",
            "positive",
            "good",
            "excellent",
            "outperform",
            "beat",
            "exceed",
            "momentum",
            "rally",
            "surge",
            "boom",
            "bull",
            "upgrade",
        }

        self.negative_words = {
            "bearish",
            "sell",
            "weak",
            "decline",
            "loss",
            "fall",
            "down",
            "negative",
            "bad",
            "poor",
            "underperform",
            "miss",
            "disappoint",
            "crash",
            "drop",
            "plunge",
            "bear",
            "downgrade",
            "risk",
        }

        self.uncertainty_words = {
            "uncertain",
            "volatile",
            "risk",
            "concern",
            "worry",
            "fear",
            "doubt",
            "question",
            "unclear",
            "unknown",
            "maybe",
            "might",
        }

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sentiment features.

        Expected columns in data:
        - news_text, social_text, analyst_reports (optional text columns)
        - news_sentiment, social_sentiment (optional pre-computed sentiment scores)
        """
        features = pd.DataFrame(index=data.index)

        # Process text-based sentiment if available
        text_columns = ["news_text", "social_text", "analyst_reports"]
        for col in text_columns:
            if col in data.columns:
                sentiment_scores = data[col].apply(self._calculate_text_sentiment)
                features[f"{col}_sentiment"] = sentiment_scores

                # Sentiment momentum
                for window in self.config.sentiment_windows:
                    features[f"{col}_sentiment_ma_{window}"] = sentiment_scores.rolling(
                        window
                    ).mean()
                    features[f"{col}_sentiment_std_{window}"] = (
                        sentiment_scores.rolling(window).std()
                    )

        # Process pre-computed sentiment scores
        sentiment_columns = ["news_sentiment", "social_sentiment", "analyst_sentiment"]
        for col in sentiment_columns:
            if col in data.columns:
                sentiment = data[col].fillna(0)

                # Rolling statistics
                for window in self.config.sentiment_windows:
                    features[f"{col}_ma_{window}"] = sentiment.rolling(window).mean()
                    features[f"{col}_std_{window}"] = sentiment.rolling(window).std()
                    features[f"{col}_min_{window}"] = sentiment.rolling(window).min()
                    features[f"{col}_max_{window}"] = sentiment.rolling(window).max()

                # Sentiment momentum and changes
                features[f"{col}_momentum_1d"] = sentiment.diff(1)
                features[f"{col}_momentum_3d"] = sentiment.diff(3)
                features[f"{col}_momentum_7d"] = sentiment.diff(7)

                # Sentiment extremes
                features[f"{col}_extreme_positive"] = (
                    sentiment > sentiment.quantile(0.9)
                ).astype(int)
                features[f"{col}_extreme_negative"] = (
                    sentiment < sentiment.quantile(0.1)
                ).astype(int)

        # News volume and attention metrics
        if "news_count" in data.columns:
            news_count = data["news_count"].fillna(0)
            for window in self.config.sentiment_windows:
                features[f"news_volume_{window}"] = news_count.rolling(window).sum()
                features[f"news_attention_{window}"] = (
                    news_count / news_count.rolling(window * 4).mean()
                ).fillna(1)

        # Social media metrics
        social_metrics = ["mentions", "likes", "shares", "comments"]
        for metric in social_metrics:
            if metric in data.columns:
                values = data[metric].fillna(0)
                for window in self.config.sentiment_windows:
                    features[f"social_{metric}_{window}"] = values.rolling(window).sum()
                    features[f"social_{metric}_growth_{window}"] = (
                        values.rolling(window).sum().pct_change()
                    )

        # Sentiment divergence (news vs social vs analyst)
        sentiment_cols = [col for col in features.columns if "sentiment_ma_7" in col]
        if len(sentiment_cols) >= 2:
            # Calculate pairwise correlations and divergences
            for i, col1 in enumerate(sentiment_cols):
                for col2 in sentiment_cols[i + 1 :]:
                    corr_name = (
                        f"sentiment_corr_{col1.split('_')[0]}_{col2.split('_')[0]}"
                    )
                    features[corr_name] = (
                        features[col1].rolling(30).corr(features[col2])
                    )

                    div_name = (
                        f"sentiment_div_{col1.split('_')[0]}_{col2.split('_')[0]}"
                    )
                    features[div_name] = abs(features[col1] - features[col2])

        return features

    def _calculate_text_sentiment(self, text: str) -> float:
        """Calculate sentiment score from text."""
        if pd.isna(text) or not isinstance(text, str):
            return 0.0

        # Simple word-based sentiment (in practice, use advanced NLP models)
        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)

        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        uncertainty_count = sum(1 for word in words if word in self.uncertainty_words)

        total_words = len(words)
        if total_words == 0:
            return 0.0

        # Calculate composite sentiment score
        sentiment = (
            positive_count - negative_count - 0.5 * uncertainty_count
        ) / total_words
        return np.clip(sentiment, -1, 1)


class NewsEmbeddingFeatures:
    """News embeddings and NLP-based features."""

    def __init__(self, config: AlternativeDataConfig):
        self.config = config
        self._initialize_topic_keywords()

    def _initialize_topic_keywords(self):
        """Initialize topic-specific keywords."""
        self.topic_keywords = {
            "earnings": [
                "earnings",
                "revenue",
                "profit",
                "eps",
                "guidance",
                "forecast",
            ],
            "merger": ["merger", "acquisition", "takeover", "deal", "buyout"],
            "regulatory": ["regulation", "fda", "approval", "compliance", "lawsuit"],
            "management": ["ceo", "cfo", "management", "leadership", "executive"],
            "product": ["product", "launch", "innovation", "patent", "technology"],
            "market": ["market", "competition", "share", "industry", "sector"],
        }

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate news embedding and NLP features.

        Expected columns:
        - news_embeddings (pre-computed embeddings)
        - news_text (raw text for processing)
        - news_topics (topic classifications)
        """
        features = pd.DataFrame(index=data.index)

        # Process pre-computed embeddings
        if "news_embeddings" in data.columns:
            embeddings = data["news_embeddings"]

            # Embedding statistics
            for window in [3, 7, 14]:
                # Average embeddings over time window
                avg_embeddings = embeddings.rolling(window).apply(
                    lambda x: np.mean([emb for emb in x if emb is not None], axis=0)
                    if any(emb is not None for emb in x)
                    else np.zeros(self.config.news_embedding_dim),
                    raw=False,
                )

                # Embedding similarity to recent average
                for i in range(
                    min(10, self.config.news_embedding_dim)
                ):  # Use first 10 dimensions
                    features[f"embedding_avg_{window}_dim_{i}"] = avg_embeddings.apply(
                        lambda x: x[i]
                        if isinstance(x, np.ndarray) and len(x) > i
                        else 0
                    )

        # Topic analysis from text
        if "news_text" in data.columns:
            for topic, keywords in self.topic_keywords.items():
                topic_scores = data["news_text"].apply(
                    lambda text: self._calculate_topic_score(text, keywords)
                )
                features[f"topic_{topic}_score"] = topic_scores

                # Topic momentum
                for window in [3, 7]:
                    features[f"topic_{topic}_ma_{window}"] = topic_scores.rolling(
                        window
                    ).mean()

        # Pre-computed topic classifications
        if "news_topics" in data.columns:
            # One-hot encode topics
            unique_topics = data["news_topics"].dropna().unique()
            for topic in unique_topics[:20]:  # Limit to top 20 topics
                topic_indicator = (data["news_topics"] == topic).astype(int)
                features[f"topic_indicator_{topic}"] = topic_indicator

                # Topic frequency over time
                for window in [7, 14]:
                    features[f"topic_freq_{topic}_{window}"] = topic_indicator.rolling(
                        window
                    ).sum()

        # News timing features
        if "news_timestamp" in data.columns:
            news_times = pd.to_datetime(data["news_timestamp"])

            # Time-of-day effects
            features["news_hour"] = news_times.dt.hour
            features["news_is_market_hours"] = (
                (news_times.dt.hour >= 9) & (news_times.dt.hour <= 16)
            ).astype(int)
            features["news_is_after_hours"] = (
                (news_times.dt.hour < 9) | (news_times.dt.hour > 16)
            ).astype(int)

            # Day-of-week effects
            features["news_day_of_week"] = news_times.dt.dayofweek
            features["news_is_weekend"] = (news_times.dt.dayofweek >= 5).astype(int)

        # News quality and source features
        if "news_source" in data.columns:
            # Source credibility (simplified scoring)
            credible_sources = {"reuters", "bloomberg", "wsj", "ft", "cnbc"}
            features["news_source_credible"] = data["news_source"].apply(
                lambda x: 1
                if isinstance(x, str) and x.lower() in credible_sources
                else 0
            )

        if "news_length" in data.columns:
            # Article length features
            features["news_length"] = data["news_length"].fillna(0)
            features["news_length_category"] = pd.cut(
                features["news_length"],
                bins=[0, 100, 500, 1000, float("inf")],
                labels=[0, 1, 2, 3],
            ).astype(float)

        return features

    def _calculate_topic_score(self, text: str, keywords: List[str]) -> float:
        """Calculate topic relevance score for text."""
        if pd.isna(text) or not isinstance(text, str):
            return 0.0

        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)

        keyword_count = sum(1 for word in words if word in keywords)
        total_words = len(words)

        return keyword_count / total_words if total_words > 0 else 0.0


class MacroeconomicFeatures:
    """Macroeconomic indicators and market-wide features."""

    def __init__(self, config: AlternativeDataConfig):
        self.config = config

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate macroeconomic features.

        Expected columns: Various macro indicators (vix, dxy, yield_10y, etc.)
        """
        features = pd.DataFrame(index=data.index)

        # Process each macro indicator
        for indicator in self.config.macro_indicators:
            if indicator in data.columns:
                values = data[indicator].fillna(method="ffill")

                # Level and changes
                features[f"macro_{indicator}"] = values
                features[f"macro_{indicator}_change_1d"] = values.diff(1)
                features[f"macro_{indicator}_change_5d"] = values.diff(5)
                features[f"macro_{indicator}_pct_change_1d"] = values.pct_change(1)
                features[f"macro_{indicator}_pct_change_5d"] = values.pct_change(5)

                # Moving averages and deviations
                for window in [5, 20, 50]:
                    ma = values.rolling(window).mean()
                    features[f"macro_{indicator}_ma_{window}"] = ma
                    features[f"macro_{indicator}_deviation_{window}"] = (
                        values - ma
                    ) / ma

                # Volatility
                for window in [10, 20]:
                    features[f"macro_{indicator}_vol_{window}"] = (
                        values.pct_change().rolling(window).std()
                    )

                # Percentile ranks (regime indicators)
                for window in [50, 100, 252]:
                    features[f"macro_{indicator}_percentile_{window}"] = values.rolling(
                        window
                    ).rank(pct=True)

        # Cross-indicator relationships
        available_indicators = [
            ind for ind in self.config.macro_indicators if ind in data.columns
        ]

        if len(available_indicators) >= 2:
            # Calculate correlations between indicators
            for i, ind1 in enumerate(available_indicators):
                for ind2 in available_indicators[i + 1 :]:
                    if ind1 in data.columns and ind2 in data.columns:
                        corr = data[ind1].rolling(50).corr(data[ind2])
                        features[f"macro_corr_{ind1}_{ind2}"] = corr

        # Risk-on/Risk-off indicators
        if "vix" in data.columns and "dxy" in data.columns:
            # Risk sentiment composite
            vix_norm = (data["vix"] - data["vix"].rolling(252).mean()) / data[
                "vix"
            ].rolling(252).std()
            dxy_norm = (data["dxy"] - data["dxy"].rolling(252).mean()) / data[
                "dxy"
            ].rolling(252).std()
            features["risk_off_indicator"] = (vix_norm + dxy_norm) / 2

        # Yield curve features
        yield_columns = [col for col in data.columns if "yield" in col.lower()]
        if len(yield_columns) >= 2:
            # Yield spreads
            for i, y1 in enumerate(yield_columns):
                for y2 in yield_columns[i + 1 :]:
                    spread = data[y1] - data[y2]
                    features[f"yield_spread_{y1}_{y2}"] = spread
                    features[f"yield_spread_change_{y1}_{y2}"] = spread.diff()

        # Economic surprise index (if available)
        if "economic_surprise" in data.columns:
            surprise = data["economic_surprise"].fillna(0)
            features["econ_surprise"] = surprise
            features["econ_surprise_ma_10"] = surprise.rolling(10).mean()
            features["econ_surprise_momentum"] = surprise.diff(5)

        return features


class SocialMediaFeatures:
    """Social media sentiment and engagement features."""

    def __init__(self, config: AlternativeDataConfig):
        self.config = config

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate social media features.

        Expected columns: Platform-specific metrics (twitter_sentiment, reddit_mentions, etc.)
        """
        features = pd.DataFrame(index=data.index)

        # Process each social platform
        for platform in self.config.social_platforms:
            # Sentiment metrics
            sentiment_col = f"{platform}_sentiment"
            if sentiment_col in data.columns:
                sentiment = data[sentiment_col].fillna(0)

                # Basic sentiment features
                features[f"social_{platform}_sentiment"] = sentiment
                for window in [1, 3, 7]:
                    features[f"social_{platform}_sentiment_ma_{window}"] = (
                        sentiment.rolling(window).mean()
                    )
                    features[f"social_{platform}_sentiment_std_{window}"] = (
                        sentiment.rolling(window).std()
                    )

                # Sentiment momentum
                features[f"social_{platform}_sentiment_momentum"] = sentiment.diff()

            # Volume/engagement metrics
            volume_metrics = ["mentions", "posts", "comments", "likes", "shares"]
            for metric in volume_metrics:
                col = f"{platform}_{metric}"
                if col in data.columns:
                    values = data[col].fillna(0)

                    # Raw values and growth
                    features[f"social_{platform}_{metric}"] = values
                    features[f"social_{platform}_{metric}_growth"] = values.pct_change()

                    # Rolling statistics
                    for window in [3, 7, 14]:
                        features[f"social_{platform}_{metric}_ma_{window}"] = (
                            values.rolling(window).mean()
                        )
                        features[f"social_{platform}_{metric}_sum_{window}"] = (
                            values.rolling(window).sum()
                        )

            # Engagement ratios
            if (
                f"{platform}_likes" in data.columns
                and f"{platform}_posts" in data.columns
            ):
                likes = data[f"{platform}_likes"].fillna(0)
                posts = data[f"{platform}_posts"].fillna(1)  # Avoid division by zero
                features[f"social_{platform}_engagement_ratio"] = likes / posts

        # Cross-platform analysis
        sentiment_platforms = [
            p for p in self.config.social_platforms if f"{p}_sentiment" in data.columns
        ]

        if len(sentiment_platforms) >= 2:
            # Sentiment consensus
            sentiment_values = [
                data[f"{p}_sentiment"].fillna(0) for p in sentiment_platforms
            ]
            features["social_sentiment_consensus"] = np.mean(sentiment_values, axis=0)
            features["social_sentiment_disagreement"] = np.std(sentiment_values, axis=0)

            # Platform correlations
            for i, p1 in enumerate(sentiment_platforms):
                for p2 in sentiment_platforms[i + 1 :]:
                    corr = (
                        data[f"{p1}_sentiment"]
                        .rolling(30)
                        .corr(data[f"{p2}_sentiment"])
                    )
                    features[f"social_corr_{p1}_{p2}"] = corr

        # Viral content detection
        for platform in self.config.social_platforms:
            if f"{platform}_shares" in data.columns:
                shares = data[f"{platform}_shares"].fillna(0)
                share_threshold = shares.rolling(30).quantile(0.95)
                features[f"social_{platform}_viral"] = (
                    shares > share_threshold
                ).astype(int)

        # Social momentum indicators
        for platform in self.config.social_platforms:
            if f"{platform}_mentions" in data.columns:
                mentions = data[f"{platform}_mentions"].fillna(0)

                # Mention acceleration
                mention_change = mentions.diff()
                features[f"social_{platform}_mention_acceleration"] = (
                    mention_change.diff()
                )

                # Buzz score (mentions relative to historical average)
                mention_ma = mentions.rolling(30).mean()
                features[f"social_{platform}_buzz_score"] = mentions / (mention_ma + 1)

        return features


class FundamentalDataFeatures:
    """Fundamental and earnings-related features."""

    def __init__(self, config: AlternativeDataConfig):
        self.config = config

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate fundamental data features.

        Expected columns: Fundamental metrics (pe_ratio, pb_ratio, etc.)
        """
        features = pd.DataFrame(index=data.index)

        # Process fundamental metrics
        for metric in self.config.fundamental_metrics:
            if metric in data.columns:
                values = data[metric].fillna(method="ffill")

                # Raw values
                features[f"fundamental_{metric}"] = values

                # Changes and trends
                features[f"fundamental_{metric}_change_1q"] = values.diff(
                    63
                )  # ~1 quarter
                features[f"fundamental_{metric}_change_1y"] = values.diff(
                    252
                )  # ~1 year

                # Percentile rankings
                for window in [252, 504]:  # 1 year, 2 years
                    features[f"fundamental_{metric}_percentile_{window}"] = (
                        values.rolling(window).rank(pct=True)
                    )

                # Trend analysis
                for window in [63, 126]:  # Quarter, half-year
                    trend = values.rolling(window).apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0]
                        if len(x) == window
                        else 0
                    )
                    features[f"fundamental_{metric}_trend_{window}"] = trend

        # Earnings-related features
        if "earnings_date" in data.columns:
            earnings_dates = pd.to_datetime(data["earnings_date"])
            current_date = data.index

            # Days to/from earnings
            days_to_earnings = (earnings_dates - current_date).dt.days
            features["days_to_earnings"] = days_to_earnings
            features["earnings_proximity"] = np.exp(
                -abs(days_to_earnings) / 30
            )  # Decay function

        if "earnings_surprise" in data.columns:
            surprise = data["earnings_surprise"].fillna(0)
            features["earnings_surprise"] = surprise
            features["earnings_surprise_ma_4q"] = surprise.rolling(
                4
            ).mean()  # 4 quarters

        # Valuation ratios and comparisons
        valuation_metrics = ["pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda"]
        available_valuations = [m for m in valuation_metrics if m in data.columns]

        if len(available_valuations) >= 2:
            # Relative valuation scores
            for metric in available_valuations:
                values = data[metric].fillna(method="ffill")

                # Z-score relative to history
                rolling_mean = values.rolling(252).mean()
                rolling_std = values.rolling(252).std()
                features[f"fundamental_{metric}_zscore"] = (values - rolling_mean) / (
                    rolling_std + 1e-8
                )

        # Growth metrics
        growth_metrics = ["revenue_growth", "earnings_growth", "book_value_growth"]
        for metric in growth_metrics:
            if metric in data.columns:
                growth = data[metric].fillna(0)
                features[f"fundamental_{metric}"] = growth

                # Growth consistency
                features[f"fundamental_{metric}_consistency"] = (
                    (growth > 0).rolling(4).sum()
                    / 4  # Fraction of positive growth quarters
                )

        # Quality scores
        if "roe" in data.columns and "debt_equity" in data.columns:
            roe = data["roe"].fillna(0)
            debt_equity = data["debt_equity"].fillna(0)

            # Simple quality score (high ROE, low debt)
            features["fundamental_quality_score"] = roe / (1 + debt_equity)

        return features


class AlternativeDataEngine:
    """Main engine for calculating all alternative data features."""

    def __init__(self, config: AlternativeDataConfig = None):
        self.config = config or AlternativeDataConfig()
        self.feature_calculators = {
            "sentiment": SentimentFeatures(self.config),
            "news_embedding": NewsEmbeddingFeatures(self.config),
            "macro": MacroeconomicFeatures(self.config),
            "social": SocialMediaFeatures(self.config),
            "fundamental": FundamentalDataFeatures(self.config),
        }

    def calculate_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all alternative data features."""
        all_features = pd.DataFrame(index=data.index)

        for feature_type, calculator in self.feature_calculators.items():
            try:
                features = calculator.calculate(data)
                # Add prefix to avoid column name conflicts
                features.columns = [
                    f"alt_{feature_type}_{col}" for col in features.columns
                ]
                all_features = pd.concat([all_features, features], axis=1)
            except Exception as e:
                print(f"Error calculating {feature_type} features: {e}")
                continue

        return all_features

    def calculate_subset(
        self, data: pd.DataFrame, feature_types: List[str]
    ) -> pd.DataFrame:
        """Calculate only specified feature types."""
        features = pd.DataFrame(index=data.index)

        for feature_type in feature_types:
            if feature_type in self.feature_calculators:
                calc_features = self.feature_calculators[feature_type].calculate(data)
                calc_features.columns = [
                    f"alt_{feature_type}_{col}" for col in calc_features.columns
                ]
                features = pd.concat([features, calc_features], axis=1)

        return features

    def get_data_requirements(self) -> Dict[str, List[str]]:
        """Get data requirements for each feature type."""
        return {
            "sentiment": [
                "news_text",
                "social_text",
                "news_sentiment",
                "social_sentiment",
            ],
            "news_embedding": [
                "news_embeddings",
                "news_text",
                "news_topics",
                "news_timestamp",
            ],
            "macro": self.config.macro_indicators,
            "social": [
                f"{p}_{m}"
                for p in self.config.social_platforms
                for m in ["sentiment", "mentions", "posts", "likes"]
            ],
            "fundamental": self.config.fundamental_metrics
            + ["earnings_date", "earnings_surprise"],
        }
