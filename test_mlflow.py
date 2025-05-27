#!/usr/bin/env python3

import mlflow
import sys
import traceback

def test_mlflow_connection():
    try:
        print("🧪 Testing MLflow HTTP server connection...")
        
        # Set tracking URI to HTTP server
        mlflow.set_tracking_uri('http://localhost:5000')
        tracking_uri = mlflow.get_tracking_uri()
        print(f"✅ MLflow tracking URI: {tracking_uri}")
        
        # Create test experiment
        experiment_name = 'notebook-integration-test'
        mlflow.set_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        print(f"✅ Test experiment: {experiment.name} (ID: {experiment.experiment_id})")
        
        # Test a simple run
        with mlflow.start_run(run_name="http_server_test"):
            mlflow.log_param('test_param', 'http_server_integration')
            mlflow.log_metric('test_metric', 0.95)
            mlflow.log_metric('server_port', 5000)
            
            run = mlflow.active_run()
            print(f"✅ Test run created: {run.info.run_id}")
            
        print("\n🎉 All MLflow HTTP server tests PASSED!")
        print("📊 Integration is working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error testing MLflow: {e}")
        print("Traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mlflow_connection()
    sys.exit(0 if success else 1)
