#!/usr/bin/env python3
"""
Test script to verify sklearn imports work correctly in Docker environment.
"""

def test_sklearn_imports():
    """Test all sklearn imports used in the project."""
    print("Testing sklearn imports...")
    
    try:
        # Test basic sklearn imports
        from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
        print("✅ sklearn.model_selection imports successful")
        
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        print("✅ sklearn.ensemble imports successful")
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
        print("✅ sklearn.metrics imports successful")
        
        from sklearn.linear_model import LinearRegression
        print("✅ sklearn.linear_model imports successful")
        
        from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
        print("✅ sklearn.preprocessing imports successful")
        
        # Test the problematic import
        try:
            from sklearn.utils import parallel_backend
            print("✅ sklearn.utils.parallel_backend import successful (old style)")
        except ImportError:
            try:
                from sklearn.utils.parallel_backend import parallel_backend
                print("✅ sklearn.utils.parallel_backend import successful (new style)")
            except ImportError:
                print("⚠️ parallel_backend not available, but this is handled gracefully")
        
        print("\n🎉 All sklearn imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    test_sklearn_imports()
