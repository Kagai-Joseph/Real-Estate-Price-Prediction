#!/usr/bin/env python3
"""
Complete Real Estate Data Pipeline and Analysis System

This module provides comprehensive data analysis, feature importance analysis,
and dataset management for the real estate price prediction system.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import category_encoders as ce
import warnings
warnings.filterwarnings('ignore')

class RealEstateDataAnalyzer:
    """
    Comprehensive analysis system for real estate datasets and price predictions.
    """
    
    def __init__(self, dataset_path='housing.csv'):
        """Initialize the analyzer with dataset path."""
        self.dataset_path = dataset_path
        self.data = None
        self.processed_data = None
        self.model = None
        self.encoder = None
        self.feature_importance = None
        self.price_influences = None
        
        # Load data if available
        if os.path.exists(dataset_path):
            self.load_dataset()
    
    def load_dataset(self):
        """Load and prepare the housing dataset."""
        print("Loading real estate dataset...")
        self.data = pd.read_csv(self.dataset_path)
        
        # Create engineered features
        self.data['rooms_per_household'] = self.data['total_rooms'] / self.data['households']
        self.data['bedrooms_per_room'] = self.data['total_bedrooms'] / self.data['total_rooms']
        self.data['population_per_household'] = self.data['population'] / self.data['households']
        self.data['log_median_income'] = np.log1p(self.data['median_income'])
        
        # Remove any NaN values
        self.data = self.data.dropna()
        
        print(f"Dataset loaded successfully: {len(self.data)} properties")
        return self.data
    
    def get_dataset_overview(self):
        """Generate comprehensive dataset overview."""
        if self.data is None:
            return "No dataset loaded"
        
        overview = {
            'basic_info': {
                'total_properties': len(self.data),
                'total_features': len(self.data.columns),
                'memory_usage': f"{self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                'date_analyzed': datetime.now().isoformat()
            },
            'numerical_features': {
                'count': len(self.data.select_dtypes(include=[np.number]).columns),
                'features': list(self.data.select_dtypes(include=[np.number]).columns)
            },
            'categorical_features': {
                'count': len(self.data.select_dtypes(include=['object']).columns),
                'features': list(self.data.select_dtypes(include=['object']).columns)
            },
            'missing_values': self.data.isnull().sum().to_dict(),
            'data_types': self.data.dtypes.astype(str).to_dict()
        }
        
        return overview
    
    def analyze_property_attributes(self):
        """Analyze residential property attributes in detail."""
        if self.data is None:
            return "No dataset loaded"
        
        attributes_analysis = {
            'location_analysis': {
                'latitude_range': {
                    'min': float(self.data['latitude'].min()),
                    'max': float(self.data['latitude'].max()),
                    'mean': float(self.data['latitude'].mean())
                },
                'longitude_range': {
                    'min': float(self.data['longitude'].min()),
                    'max': float(self.data['longitude'].max()),
                    'mean': float(self.data['longitude'].mean())
                },
                'ocean_proximity_distribution': self.data['ocean_proximity'].value_counts().to_dict()
            },
            'housing_characteristics': {
                'median_age_stats': {
                    'min': float(self.data['housing_median_age'].min()),
                    'max': float(self.data['housing_median_age'].max()),
                    'mean': float(self.data['housing_median_age'].mean()),
                    'median': float(self.data['housing_median_age'].median())
                },
                'room_statistics': {
                    'total_rooms_avg': float(self.data['total_rooms'].mean()),
                    'total_bedrooms_avg': float(self.data['total_bedrooms'].mean()),
                    'rooms_per_household_avg': float(self.data['rooms_per_household'].mean()),
                    'bedrooms_per_room_avg': float(self.data['bedrooms_per_room'].mean())
                }
            },
            'demographic_data': {
                'population_stats': {
                    'min': float(self.data['population'].min()),
                    'max': float(self.data['population'].max()),
                    'mean': float(self.data['population'].mean()),
                    'median': float(self.data['population'].median())
                },
                'household_stats': {
                    'min': float(self.data['households'].min()),
                    'max': float(self.data['households'].max()),
                    'mean': float(self.data['households'].mean()),
                    'population_per_household_avg': float(self.data['population_per_household'].mean())
                },
                'income_analysis': {
                    'median_income_min': float(self.data['median_income'].min()),
                    'median_income_max': float(self.data['median_income'].max()),
                    'median_income_avg': float(self.data['median_income'].mean()),
                    'high_income_areas': len(self.data[self.data['median_income'] > 10])
                }
            }
        }
        
        return attributes_analysis
    
    def analyze_selling_prices(self):
        """Analyze residential property selling prices."""
        if self.data is None:
            return "No dataset loaded"
        
        # Handle missing values safely
        price_col = 'median_house_value'
        if price_col not in self.data.columns:
            return {"error": "median_house_value column not found"}
        
        price_data = self.data[price_col].dropna()
        
        price_analysis = {
            'price_statistics': {
                'min_price': float(price_data.min()),
                'max_price': float(price_data.max()),
                'mean_price': float(price_data.mean()),
                'median_price': float(price_data.median()),
                'std_price': float(price_data.std()),
                'total_value': float(price_data.sum())
            },
            'price_distribution': {
                'under_200k': len(self.data[price_data < 200000]),
                '200k_to_400k': len(self.data[(price_data >= 200000) & (price_data < 400000)]),
                '400k_to_600k': len(self.data[(price_data >= 400000) & (price_data < 600000)]),
                'over_600k': len(self.data[price_data >= 600000])
            },
            'price_by_location': {},
            'price_correlations': {}
        }
        
        # Safely calculate price by location
        try:
            coastal_properties = self.data[self.data['ocean_proximity'].isin(['NEAR OCEAN', 'NEAR BAY'])]
            inland_properties = self.data[self.data['ocean_proximity'] == 'INLAND']
            ocean_1h_properties = self.data[self.data['ocean_proximity'] == '<1H OCEAN']
            
            price_analysis['price_by_location'] = {
                'coastal_avg': float(coastal_properties[price_col].mean()) if len(coastal_properties) > 0 else 0,
                'inland_avg': float(inland_properties[price_col].mean()) if len(inland_properties) > 0 else 0,
                'ocean_1h_avg': float(ocean_1h_properties[price_col].mean()) if len(ocean_1h_properties) > 0 else 0
            }
        except Exception as e:
            print(f"Error calculating price by location: {e}")
            price_analysis['price_by_location'] = {
                'coastal_avg': 0,
                'inland_avg': 0,
                'ocean_1h_avg': 0
            }
        
        # Calculate correlations for numerical columns only
        try:
            numerical_data = self.data.select_dtypes(include=[np.number])
            if price_col in numerical_data.columns:
                correlations = numerical_data.corr()[price_col].sort_values(ascending=False)
                price_analysis['price_correlations'] = correlations.to_dict()
        except Exception as e:
            print(f"Error calculating correlations: {e}")
            price_analysis['price_correlations'] = {}
        
        return price_analysis
    
    def identify_key_price_influences(self):
        """Identify and analyze key factors that influence property prices."""
        if self.data is None:
            return "No dataset loaded"
        
        try:
            # Prepare features for analysis
            features = self.data.drop(columns=['median_house_value'])
            target = self.data['median_house_value']
            
            # Handle categorical variables with target encoding
            encoder = ce.TargetEncoder(cols=['ocean_proximity'])
            features_encoded = encoder.fit_transform(features, target)
            
            # Train a Random Forest for feature importance
            rf_analyzer = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_analyzer.fit(features_encoded, target)
            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'feature': features_encoded.columns,
                'importance': rf_analyzer.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Calculate correlation with price for numerical features only
            correlations = features_encoded.corrwith(target).abs().sort_values(ascending=False)
            
            # Safely calculate income quartile prices
            income_quartile_prices = {}
            try:
                income_quartiles = pd.qcut(self.data['median_income'], 4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
                income_quartile_prices = self.data.groupby(income_quartiles)['median_house_value'].mean().to_dict()
            except Exception as e:
                print(f"Error calculating income quartiles: {e}")
                income_quartile_prices = {'Low': 0, 'Medium-Low': 0, 'Medium-High': 0, 'High': 0}
            
            # Analyze price impact by categories
            price_influences = {
                'feature_importance_ranking': feature_importance.to_dict('records'),
                'correlation_analysis': {k: float(v) for k, v in correlations.items() if pd.notna(v)},
                'top_5_features': feature_importance.head(5)['feature'].tolist(),
                'location_impact': {
                    'ocean_proximity_effect': self.data.groupby('ocean_proximity')['median_house_value'].mean().to_dict(),
                    'latitude_correlation': float(correlations.get('latitude', 0)),
                    'longitude_correlation': float(correlations.get('longitude', 0))
                },
                'income_impact': {
                    'income_correlation': float(correlations.get('median_income', 0)),
                    'log_income_correlation': float(correlations.get('log_median_income', 0)),
                    'income_quartile_prices': income_quartile_prices
                },
                'housing_characteristics_impact': {
                    'age_correlation': float(correlations.get('housing_median_age', 0)),
                    'rooms_correlation': float(correlations.get('total_rooms', 0)),
                    'rooms_per_household_correlation': float(correlations.get('rooms_per_household', 0)),
                    'bedrooms_per_room_correlation': float(correlations.get('bedrooms_per_room', 0))
                },
                'demographic_impact': {
                    'population_correlation': float(correlations.get('population', 0)),
                    'households_correlation': float(correlations.get('households', 0)),
                    'population_per_household_correlation': float(correlations.get('population_per_household', 0))
                }
            }
            
            self.feature_importance = feature_importance
            self.price_influences = price_influences
            
            return price_influences
            
        except Exception as e:
            print(f"Error in identify_key_price_influences: {e}")
            return {
                'feature_importance_ranking': [],
                'correlation_analysis': {},
                'top_5_features': [],
                'location_impact': {'ocean_proximity_effect': {}, 'latitude_correlation': 0, 'longitude_correlation': 0},
                'income_impact': {'income_correlation': 0, 'log_income_correlation': 0, 'income_quartile_prices': {}},
                'housing_characteristics_impact': {'age_correlation': 0, 'rooms_correlation': 0, 'rooms_per_household_correlation': 0, 'bedrooms_per_room_correlation': 0},
                'demographic_impact': {'population_correlation': 0, 'households_correlation': 0, 'population_per_household_correlation': 0}
            }
    
    def generate_market_insights(self):
        """Generate comprehensive market insights and trends."""
        if self.data is None:
            return "No dataset loaded"
        
        try:
            price_col = 'median_house_value'
            
            insights = {
                'market_overview': {
                    'total_properties_analyzed': len(self.data),
                    'total_market_value': float(self.data[price_col].sum()),
                    'average_property_value': float(self.data[price_col].mean()),
                    'market_diversity_score': float(self.data[price_col].std() / self.data[price_col].mean())
                },
                'premium_markets': {
                    'top_10_percent_threshold': float(self.data[price_col].quantile(0.9)),
                    'premium_properties_count': len(self.data[self.data[price_col] > self.data[price_col].quantile(0.9)]),
                    'premium_market_characteristics': {}
                },
                'affordable_markets': {
                    'bottom_25_percent_threshold': float(self.data[price_col].quantile(0.25)),
                    'affordable_properties_count': len(self.data[self.data[price_col] < self.data[price_col].quantile(0.25)]),
                    'affordable_market_characteristics': {}
                },
                'investment_opportunities': {
                    'high_room_density_areas': len(self.data[self.data['rooms_per_household'] > self.data['rooms_per_household'].quantile(0.8)]),
                    'growing_population_areas': len(self.data[self.data['population_per_household'] > 3]),
                    'undervalued_high_income_areas': len(self.data[(self.data['median_income'] > self.data['median_income'].quantile(0.7)) & 
                                                                 (self.data[price_col] < self.data[price_col].median())])
                }
            }
            
            # Safely calculate premium market characteristics
            try:
                premium_properties = self.data[self.data[price_col] > self.data[price_col].quantile(0.9)]
                insights['premium_markets']['premium_market_characteristics'] = {
                    'avg_income': float(premium_properties['median_income'].mean()),
                    'avg_age': float(premium_properties['housing_median_age'].mean()),
                    'ocean_proximity_dist': premium_properties['ocean_proximity'].value_counts().to_dict()
                }
            except Exception as e:
                print(f"Error calculating premium market characteristics: {e}")
                insights['premium_markets']['premium_market_characteristics'] = {
                    'avg_income': 0,
                    'avg_age': 0,
                    'ocean_proximity_dist': {}
                }
            
            # Safely calculate affordable market characteristics
            try:
                affordable_properties = self.data[self.data[price_col] < self.data[price_col].quantile(0.25)]
                insights['affordable_markets']['affordable_market_characteristics'] = {
                    'avg_income': float(affordable_properties['median_income'].mean()),
                    'avg_age': float(affordable_properties['housing_median_age'].mean()),
                    'ocean_proximity_dist': affordable_properties['ocean_proximity'].value_counts().to_dict()
                }
            except Exception as e:
                print(f"Error calculating affordable market characteristics: {e}")
                insights['affordable_markets']['affordable_market_characteristics'] = {
                    'avg_income': 0,
                    'avg_age': 0,
                    'ocean_proximity_dist': {}
                }
            
            return insights
            
        except Exception as e:
            print(f"Error in generate_market_insights: {e}")
            return {
                'market_overview': {'total_properties_analyzed': 0, 'total_market_value': 0, 'average_property_value': 0, 'market_diversity_score': 0},
                'premium_markets': {'top_10_percent_threshold': 0, 'premium_properties_count': 0, 'premium_market_characteristics': {}},
                'affordable_markets': {'bottom_25_percent_threshold': 0, 'affordable_properties_count': 0, 'affordable_market_characteristics': {}},
                'investment_opportunities': {'high_room_density_areas': 0, 'growing_population_areas': 0, 'undervalued_high_income_areas': 0}
            }
    
    def create_predictive_model_analysis(self):
        """Analyze the predictive model performance and characteristics."""
        if self.data is None:
            return "No dataset loaded"
        
        # Load existing model if available
        model_path = './models/random_forest_regressor_model.pkl'
        encoder_path = './models/target_encoder.pkl'
        
        model_analysis = {
            'model_availability': {
                'model_file_exists': os.path.exists(model_path),
                'encoder_file_exists': os.path.exists(encoder_path),
                'last_modified': None
            }
        }
        
        if os.path.exists(model_path):
            try:
                # Get file modification time
                model_analysis['model_availability']['last_modified'] = datetime.fromtimestamp(
                    os.path.getmtime(model_path)
                ).isoformat()
                
                # Load model and perform analysis
                model = joblib.load(model_path)
                encoder = joblib.load(encoder_path)
                
                # Prepare data for model evaluation
                X = self.data.drop(columns=['median_house_value'])
                y = self.data['median_house_value']
                
                # Apply same preprocessing as in training
                X_encoded = encoder.transform(X)
                
                # Make predictions
                predictions = model.predict(X_encoded)
                
                # Calculate metrics
                mae = mean_absolute_error(y, predictions)
                mse = mean_squared_error(y, predictions)
                rmse = np.sqrt(mse)
                r2 = r2_score(y, predictions)
                
                model_analysis.update({
                    'model_performance': {
                        'r2_score': float(r2),
                        'mean_absolute_error': float(mae),
                        'mean_squared_error': float(mse),
                        'root_mean_squared_error': float(rmse),
                        'accuracy_percentage': float((1 - mae/y.mean()) * 100) if y.mean() > 0 else 0
                    },
                    'prediction_analysis': {
                        'prediction_range': {
                            'min': float(predictions.min()),
                            'max': float(predictions.max()),
                            'mean': float(predictions.mean())
                        },
                        'residual_analysis': {
                            'mean_residual': float((y - predictions).mean()),
                            'residual_std': float((y - predictions).std())
                        }
                    },
                    'model_characteristics': {
                        'model_type': str(type(model).__name__),
                        'n_estimators': getattr(model, 'n_estimators', 'N/A'),
                        'feature_count': len(X_encoded.columns) if hasattr(X_encoded, 'columns') else 'N/A'
                    }
                })
                
            except Exception as e:
                print(f"Error in model analysis: {e}")
                model_analysis['model_analysis_error'] = str(e)
        
        return model_analysis

def main():
    """Main function to run comprehensive analysis."""
    print("=== Real Estate Data Analysis System ===")
    
    # Initialize analyzer
    analyzer = RealEstateDataAnalyzer('housing.csv')
    
    if analyzer.data is None:
        print("Error: Could not load housing dataset. Please ensure 'housing.csv' exists.")
        return
    
    print(f"Loaded dataset with {len(analyzer.data)} properties")
    
    # Run comprehensive analysis
    print("\n1. Analyzing dataset overview...")
    overview = analyzer.get_dataset_overview()
    print(f"   - Total properties: {overview['basic_info']['total_properties']}")
    print(f"   - Total features: {overview['basic_info']['total_features']}")
    
    print("\n2. Analyzing property attributes...")
    attributes = analyzer.analyze_property_attributes()
    print(f"   - Ocean proximity types: {len(attributes['location_analysis']['ocean_proximity_distribution'])}")
    print(f"   - Average rooms per household: {attributes['housing_characteristics']['room_statistics']['rooms_per_household_avg']:.2f}")
    
    print("\n3. Analyzing selling prices...")
    prices = analyzer.analyze_selling_prices()
    print(f"   - Average house value: ${prices['price_statistics']['mean_price']:,.2f}")
    print(f"   - Price range: ${prices['price_statistics']['min_price']:,.2f} - ${prices['price_statistics']['max_price']:,.2f}")
    
    print("\n4. Identifying key price influences...")
    influences = analyzer.identify_key_price_influences()
    print(f"   - Top 3 most important features:")
    for i, feature in enumerate(influences['top_5_features'][:3], 1):
        print(f"     {i}. {feature}")
    
    print("\n5. Generating market insights...")
    insights = analyzer.generate_market_insights()
    print(f"   - Total market value: ${insights['market_overview']['total_market_value']:,.2f}")
    print(f"   - Premium properties (top 10%): {insights['premium_markets']['premium_properties_count']}")
    
    print("\n6. Analyzing predictive model...")
    model_analysis = analyzer.create_predictive_model_analysis()
    if model_analysis['model_availability']['model_file_exists']:
        if 'model_performance' in model_analysis:
            print(f"   - Model R² score: {model_analysis['model_performance']['r2_score']:.3f}")
            print(f"   - Model accuracy: {model_analysis['model_performance']['accuracy_percentage']:.1f}%")
        else:
            print("   - Model exists but analysis failed")
    else:
        print("   - No trained model found")
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()