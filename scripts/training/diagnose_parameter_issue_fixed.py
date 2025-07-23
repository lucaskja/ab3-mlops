#!/usr/bin/env python3
"""
Fixed diagnostic script to show the parameter access issue and solution
"""

from sagemaker.workflow.parameters import ParameterString, ParameterInteger, ParameterFloat

def demonstrate_parameter_issue():
    """Demonstrate the parameter access issue you were experiencing"""
    
    print("🔍 Diagnosing SageMaker Pipeline Parameter Access Issue")
    print("=" * 60)
    
    # 1. Create parameters as you did (as a list)
    print("1️⃣ Creating parameters as a list (correct for Pipeline constructor):")
    
    parameters = [
        ParameterString(name="DatasetPath", default_value="s3://bucket/data"),
        ParameterString(name="DatasetName", default_value="drone-detection"),
        ParameterString(name="ModelVariant", default_value="yolov11n"),
        ParameterInteger(name="BatchSize", default_value=16),
        ParameterFloat(name="LearningRate", default_value=0.001)
    ]
    
    print(f"   ✅ Created {len(parameters)} parameters")
    for param in parameters:
        print(f"      - {param.name}: {param.default_value}")
    
    # 2. Convert to dictionary for step access (your approach)
    print("\n2️⃣ Converting to dictionary for step access:")
    
    param_dict = {p.name: p for p in parameters}
    
    print(f"   ✅ Dictionary keys: {list(param_dict.keys())}")
    
    # 3. Show the WRONG access pattern (what was causing the error)
    print("\n3️⃣ WRONG access pattern (causes KeyError):")
    
    wrong_keys = ['dataset_path', 'dataset_name', 'model_variant', 'batch_size', 'learning_rate']
    
    for wrong_key in wrong_keys:
        try:
            value = param_dict[wrong_key]
            print(f"   ✅ param_dict['{wrong_key}'] exists")
        except KeyError as e:
            print(f"   ❌ KeyError: param_dict['{wrong_key}'] - {e}")
    
    # 4. Show the CORRECT access pattern
    print("\n4️⃣ CORRECT access pattern (works):")
    
    correct_keys = ['DatasetPath', 'DatasetName', 'ModelVariant', 'BatchSize', 'LearningRate']
    
    for correct_key in correct_keys:
        try:
            value = param_dict[correct_key]
            # Use .expr to display parameter info without triggering __str__ error
            print(f"   ✅ param_dict['{correct_key}'] exists (type: {type(value).__name__})")
        except KeyError as e:
            print(f"   ❌ KeyError: param_dict['{correct_key}'] - {e}")
    
    # 5. Show the specific error from your code
    print("\n5️⃣ Your specific error analysis:")
    
    print("   In your create_evaluation_step function, you had:")
    print("   ```python")
    print("   source=parameters['dataset_path'],  # ❌ WRONG - 'dataset_path' key doesn't exist")
    print("   ```")
    print("   ")
    print("   But 'dataset_path' doesn't exist in param_dict. The available keys are:")
    for key in param_dict.keys():
        print(f"      - '{key}'")
    print("   ")
    print("   The correct access should be:")
    print("   ```python")
    print("   source=param_dict['DatasetPath'],   # ✅ CORRECT - matches parameter.name")
    print("   ```")
    
    # 6. Show the solution
    print("\n6️⃣ Solution summary:")
    print("   ✅ Parameter dictionary keys are the 'name' attribute from parameter creation")
    print("   ✅ Use param_dict[parameter.name] not param_dict[your_variable_name]")
    print("   ✅ Parameter names are case-sensitive (DatasetPath ≠ dataset_path)")
    print("   ✅ Pipeline constructor takes list of parameters, not dictionary")
    print("   ✅ SageMaker parameters are special objects, not simple strings/numbers")

def show_your_exact_error():
    """Show exactly what was happening in your code"""
    
    print("\n🚨 Your Exact Error Reproduction")
    print("=" * 35)
    
    # Simulate your parameter creation
    parameters = [
        ParameterString(name="DatasetPath", default_value="s3://bucket/data"),
        ParameterString(name="DatasetName", default_value="drone-detection")
    ]
    
    # Your conversion pattern
    param_dict = {p.name: p for p in parameters}
    
    print("Available keys in param_dict:")
    for key in param_dict.keys():
        print(f"   - '{key}'")
    
    print("\nYour code was trying to access:")
    
    # This is what was failing in your create_evaluation_step
    try:
        dataset_path = param_dict['dataset_path']  # This was your error
        print("   ✅ param_dict['dataset_path'] - SUCCESS")
    except KeyError as e:
        print(f"   ❌ param_dict['dataset_path'] - KeyError: {e}")
        print("      This is exactly the error you got!")
    
    # The correct access
    try:
        dataset_path = param_dict['DatasetPath']  # This is correct
        print("   ✅ param_dict['DatasetPath'] - SUCCESS")
    except KeyError as e:
        print(f"   ❌ param_dict['DatasetPath'] - KeyError: {e}")

def show_corrected_patterns():
    """Show the corrected code patterns"""
    
    print("\n🔧 Corrected Code Patterns")
    print("=" * 30)
    
    print("✅ Parameter Creation (correct):")
    print("""
    dataset_path = ParameterString(
        name="DatasetPath",  # ← This becomes the dictionary key
        default_value="s3://bucket/data"
    )
    
    parameters = [dataset_path, ...]  # List for Pipeline constructor
    param_dict = {p.name: p for p in parameters}  # Dict for step access
    """)
    
    print("✅ Parameter Access in Steps (correct):")
    print("""
    ProcessingInput(
        source=param_dict['DatasetPath'],  # ← Use parameter.name as key
        destination="/opt/ml/processing/input"
    )
    """)
    
    print("❌ What was wrong in your code:")
    print("""
    # In create_evaluation_step function:
    ProcessingInput(
        source=parameters['dataset_path'],  # ← Wrong: 'dataset_path' key doesn't exist
        destination="/opt/ml/processing/test"
    )
    
    # Should have been:
    ProcessingInput(
        source=param_dict['DatasetPath'],   # ← Correct: matches parameter.name
        destination="/opt/ml/processing/test"
    )
    """)

if __name__ == "__main__":
    demonstrate_parameter_issue()
    show_your_exact_error()
    show_corrected_patterns()
