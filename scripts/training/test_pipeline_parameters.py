#!/usr/bin/env python3
"""
Test script to validate SageMaker Pipeline parameter structure
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from sagemaker.workflow.parameters import (
    ParameterString,
    ParameterInteger,
    ParameterFloat
)

def test_parameter_creation():
    """Test parameter creation and access patterns"""
    
    print("🧪 Testing SageMaker Pipeline Parameter Creation...")
    
    # Create parameters the same way as in the pipeline
    parameters = {
        'dataset_path': ParameterString(
            name="DatasetPath",
            default_value="s3://lucaskle-ab3-project-pv/data/training"
        ),
        'dataset_name': ParameterString(
            name="DatasetName", 
            default_value="drone-detection"
        ),
        'model_variant': ParameterString(
            name="ModelVariant",
            default_value="yolov11n"
        ),
        'batch_size': ParameterInteger(
            name="BatchSize",
            default_value=16
        ),
        'learning_rate': ParameterFloat(
            name="LearningRate",
            default_value=0.001
        )
    }
    
    print(f"✅ Created {len(parameters)} parameters")
    
    # Test parameter access
    print("\n📋 Parameter Details:")
    for key, param in parameters.items():
        print(f"   {key}: {param.name} = {param.default_value}")
    
    # Test parameter list conversion (what Pipeline expects)
    parameter_list = list(parameters.values())
    print(f"\n✅ Parameter list has {len(parameter_list)} items")
    
    # Test parameter dictionary access pattern
    print("\n🔍 Testing parameter access patterns:")
    
    # This is how we access in step creation
    try:
        dataset_path = parameters['dataset_path']
        print(f"   ✅ parameters['dataset_path'] = {dataset_path}")
        print(f"   ✅ Parameter name: {dataset_path.name}")
        print(f"   ✅ Default value: {dataset_path.default_value}")
    except KeyError as e:
        print(f"   ❌ KeyError accessing 'dataset_path': {e}")
    
    # Test all parameter access
    test_keys = ['dataset_path', 'dataset_name', 'model_variant', 'batch_size', 'learning_rate']
    
    for key in test_keys:
        try:
            param = parameters[key]
            print(f"   ✅ {key}: {param.name}")
        except KeyError as e:
            print(f"   ❌ KeyError accessing '{key}': {e}")
    
    return parameters

def test_parameter_dict_conversion():
    """Test the parameter dictionary conversion pattern from your summary"""
    
    print("\n🔄 Testing parameter dictionary conversion pattern...")
    
    # Simulate the pattern from your conversation summary
    parameters_list = [
        ParameterString(name="DatasetPath", default_value="s3://test/path"),
        ParameterString(name="DatasetName", default_value="test-dataset"),
        ParameterString(name="ModelVariant", default_value="yolov11n")
    ]
    
    # Convert to dictionary (the pattern that was causing issues)
    param_dict = {p.name: p for p in parameters_list}
    
    print(f"✅ Converted {len(parameters_list)} parameters to dictionary")
    print("📋 Dictionary keys:", list(param_dict.keys()))
    
    # Test access patterns
    try:
        dataset_path = param_dict['DatasetPath']
        print(f"   ✅ param_dict['DatasetPath'] = {dataset_path}")
    except KeyError as e:
        print(f"   ❌ KeyError: {e}")
    
    # Test the problematic access pattern
    try:
        dataset_path = param_dict['dataset_path']  # This would fail
        print(f"   ✅ param_dict['dataset_path'] = {dataset_path}")
    except KeyError as e:
        print(f"   ❌ Expected KeyError for 'dataset_path': {e}")
        print("   💡 This shows the naming mismatch issue!")

def main():
    """Main test function"""
    
    print("🚀 SageMaker Pipeline Parameter Testing")
    print("=" * 50)
    
    # Test 1: Parameter creation
    parameters = test_parameter_creation()
    
    # Test 2: Dictionary conversion pattern
    test_parameter_dict_conversion()
    
    print("\n✅ All parameter tests completed!")
    print("\n💡 Key Insights:")
    print("   - Use consistent naming between parameter creation and access")
    print("   - Parameter objects have .name and .default_value attributes")
    print("   - Pipeline expects list of parameter objects, not dictionary")
    print("   - Access parameters by the key used in creation, not parameter.name")

if __name__ == "__main__":
    main()
