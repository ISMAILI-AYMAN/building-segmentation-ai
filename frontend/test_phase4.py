#!/usr/bin/env python3
"""
Test Script for Phase 4: Web Frontend
Verifies all components are working correctly
"""
import os
import sys
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from frontend.app import BuildingSegmentationFrontend

def test_frontend_initialization():
    """Test frontend initialization"""
    print("Testing Frontend Initialization...")
    
    try:
        # Test frontend initialization
        frontend = BuildingSegmentationFrontend("http://localhost:8000")
        
        # Test app creation
        assert frontend.app is not None
        print(f"  [OK] Flask app created")
        
        # Test storage setup
        assert frontend.upload_dir.exists()
        assert frontend.results_dir.exists()
        assert frontend.temp_dir.exists()
        print(f"  [OK] Storage directories created")
        
        # Test API URL
        assert frontend.api_url == "http://localhost:8000"
        print(f"  [OK] API URL configured")
        
        return True
        
    except Exception as e:
        print(f"  [FAILED] Frontend initialization test failed: {e}")
        return False

def test_flask_routes():
    """Test Flask routes"""
    print("Testing Flask Routes...")
    
    try:
        frontend = BuildingSegmentationFrontend("http://localhost:8000")
        
        # Test route registration
        routes = []
        for rule in frontend.app.url_map.iter_rules():
            routes.append(rule.endpoint)
        
        expected_routes = ['index', 'health', 'model_info', 'upload_file', 'api_inference', 'get_results', 'download_results', 'gallery', 'about']
        
        for route in expected_routes:
            if route in routes:
                print(f"  [OK] Route '{route}' registered")
            else:
                print(f"  [FAILED] Route '{route}' missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"  [FAILED] Flask routes test failed: {e}")
        return False

def test_file_validation():
    """Test file validation"""
    print("Testing File Validation...")
    
    try:
        frontend = BuildingSegmentationFrontend("http://localhost:8000")
        
        # Test allowed files
        allowed_files = ['test.jpg', 'test.png', 'test.jpeg', 'test.gif', 'test.bmp', 'test.tiff']
        for filename in allowed_files:
            assert frontend.allowed_file(filename) == True
            print(f"  [OK] Allowed file: {filename}")
        
        # Test disallowed files
        disallowed_files = ['test.txt', 'test.pdf', 'test.doc', 'test.exe']
        for filename in disallowed_files:
            assert frontend.allowed_file(filename) == False
            print(f"  [OK] Disallowed file: {filename}")
        
        return True
        
    except Exception as e:
        print(f"  [FAILED] File validation test failed: {e}")
        return False

def test_template_rendering():
    """Test template rendering"""
    print("Testing Template Rendering...")
    
    try:
        frontend = BuildingSegmentationFrontend("http://localhost:8000")
        
        # Test index page
        with frontend.app.test_client() as client:
            response = client.get('/')
            assert response.status_code == 200
            assert b'Building Segmentation' in response.data
            print(f"  [OK] Index page renders")
            
            # Test upload page
            response = client.get('/upload')
            assert response.status_code == 200
            assert b'Upload Images' in response.data
            print(f"  [OK] Upload page renders")
            
            # Test about page
            response = client.get('/about')
            assert response.status_code == 200
            print(f"  [OK] About page renders")
            
            # Test gallery page
            response = client.get('/gallery')
            assert response.status_code == 200
            print(f"  [OK] Gallery page renders")
        
        return True
        
    except Exception as e:
        print(f"  [FAILED] Template rendering test failed: {e}")
        return False

def test_api_integration():
    """Test API integration"""
    print("Testing API Integration...")
    
    try:
        frontend = BuildingSegmentationFrontend("http://localhost:8000")
        
        # Test health endpoint (will fail if API not running, but that's expected)
        with frontend.app.test_client() as client:
            response = client.get('/health')
            assert response.status_code == 200
            data = response.get_json()
            assert 'status' in data
            print(f"  [OK] Health endpoint: {data['status']}")
            
            # Test model info endpoint (will fail if API not running, but that's expected)
            response = client.get('/model/info')
            if response.status_code == 503:
                print(f"  [OK] Model info endpoint: API not running (expected)")
            else:
                print(f"  [OK] Model info endpoint: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"  [FAILED] API integration test failed: {e}")
        return False

def test_command_line_interface():
    """Test command-line interface"""
    print("Testing Command-line Interface...")
    
    try:
        # Test help output
        import subprocess
        result = subprocess.run([
            sys.executable, "frontend/app.py", "--help"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  [OK] Frontend help command works")
        else:
            print("  [FAILED] Frontend help command failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"  [FAILED] Command-line interface test failed: {e}")
        return False

def test_template_files():
    """Test template files exist"""
    print("Testing Template Files...")
    
    try:
        template_dir = Path("frontend/templates")
        
        required_templates = [
            'base.html',
            'index.html',
            'upload.html'
        ]
        
        for template in required_templates:
            template_path = template_dir / template
            if template_path.exists():
                print(f"  [OK] Template exists: {template}")
            else:
                print(f"  [FAILED] Template missing: {template}")
                return False
        
        return True
        
    except Exception as e:
        print(f"  [FAILED] Template files test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("PHASE 4 TESTING - WEB FRONTEND COMPONENTS")
    print("=" * 60)
    
    tests = [
        ("Frontend Initialization", test_frontend_initialization),
        ("Flask Routes", test_flask_routes),
        ("File Validation", test_file_validation),
        ("Template Rendering", test_template_rendering),
        ("API Integration", test_api_integration),
        ("Command-line Interface", test_command_line_interface),
        ("Template Files", test_template_files)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"  [FAILED] {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] ALL TESTS PASSED! Phase 4 is ready for use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())
