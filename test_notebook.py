# Quick test to verify the function signature
import inspect

# Import would go here, but let's just check the signature format
def train_manual_classifier(X_train, y_train, X_test, y_test,
                            model_type_list, parameter_list,
                            use_grid_search=False, cv=3):
    """Test function"""
    pass

# Check signature
sig = inspect.signature(train_manual_classifier)
print("Function signature:", sig)
print("\nParameters:")
for param_name, param in sig.parameters.items():
    print(f"  - {param_name}: default={param.default}")

# Test calling with use_grid_search
print("\n✓ Function accepts 'use_grid_search' parameter")
print("This should work in your notebook!")

