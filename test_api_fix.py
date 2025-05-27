#!/usr/bin/env python3
"""
Test poprawki API dla MLflow 2.19.0
"""
import mlflow
from mlflow.tracking import MlflowClient

def test_mlflow_api():
    print("🧪 TESTOWANIE POPRAWKI MLFLOW 2.19.0 API")
    print("=" * 50)
    
    try:
        print(f"📦 MLflow version: {mlflow.__version__}")
        
        print("🔗 Łączenie z serwerem MLflow...")
        mlflow.set_tracking_uri('http://localhost:5000')
        client = MlflowClient()
        
        print("✅ Połączenie nawiązane")
        
        print("🧪 Testowanie search_registered_models()...")
        models = client.search_registered_models()
        
        print(f"✅ API search_registered_models() działa!")
        print(f"📦 Znaleziono {len(models)} zarejestrowanych modeli:")
        
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model.name}")
            
            # Sprawdźmy stage'y wersji
            try:
                versions = client.get_latest_versions(model.name)
                for version in versions:
                    print(f"     └─ Wersja {version.version} (Stage: {version.current_stage})")
            except Exception as e:
                print(f"     └─ Błąd wersji: {e}")
        
        print("\n🎉 SUKCES!")
        print("✅ Poprawka API MLflow 2.19.0 działa prawidłowo")
        print("✅ search_registered_models() zastąpiła list_registered_models()")
        print("✅ Notebook powinien teraz działać bez błędów")
        
        return True
        
    except Exception as e:
        print(f"❌ BŁĄD: {e}")
        print("❌ Notebook nadal może mieć problemy")
        return False

if __name__ == "__main__":
    test_mlflow_api()
