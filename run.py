import os
import platform
import subprocess
import sys
import time
import webbrowser
import argparse
from pathlib import Path
from utils.config import Config
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)

def check_data_files():
    """Check if required data files exist and are not empty"""
    data_dir = Path("data")
    output_dir = Path("output")
    
    required_files = [
        data_dir / "customers.csv",
        data_dir / "transactions.csv",
        output_dir / "churn_data.csv",
        output_dir / "revenue_data.csv"
    ]
    
    missing_files = []
    empty_files = []
    
    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(str(file_path))
        elif file_path.stat().st_size == 0:
            empty_files.append(str(file_path))
    
    return missing_files, empty_files

def check_input_data_files():
    """Check if required input data files exist and are not empty"""
    data_dir = Path("data")
    required_files = [
        data_dir / "customers.csv",
        data_dir / "transactions.csv"
    ]
    missing_files = []
    empty_files = []
    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(str(file_path))
        elif file_path.stat().st_size == 0:
            empty_files.append(str(file_path))
    return missing_files, empty_files

def generate_sample_data():
    """Generate sample data for the dashboard"""
    try:
        print("📊 Generating sample data...")
        
        # Import and run the data generation
        from processor.clean_and_merge import clean_and_merge
        from processor.churn_model import generate_churn_data
        from processor.revenue_model import generate_revenue_data
        
        # Check if we have a data generation script
        if Path("generate_sample_data.py").exists():
            result = subprocess.run([sys.executable, "generate_sample_data.py"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Sample data generated successfully!")
                return True
            else:
                print(f"❌ Error generating sample data: {result.stderr}")
                return False
        else:
            print("❌ Sample data generation script not found")
            return False
            
    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        print(f"❌ Error generating sample data: {e}")
        return False

def run_streamlit_app():
    """Run the Streamlit dashboard"""
    try:
        print("🚀 Starting AutoInsights Streamlit Dashboard...")
        print("📊 Loading Interactive Dashboard")
        print("🌐 Dashboard will be available at: http://localhost:8501")
        print("=" * 60)
        
        # Get the path to the Streamlit app
        streamlit_app_path = Path("app.py")
        
        if not streamlit_app_path.exists():
            logger.error(f"Streamlit app not found at {streamlit_app_path}")
            print("💡 Make sure app.py exists in the project root directory")
            return False

        # Check if streamlit is installed
        try:
            import streamlit
        except ImportError:
            print("❌ Streamlit is not installed. Please install it with: pip install streamlit")
            return False

        # Prepare the Streamlit command
        if platform.system() == "Windows":
            # Use venv streamlit if available, otherwise system streamlit
            venv_streamlit = Path("venv/Scripts/streamlit.exe")
            if venv_streamlit.exists():
                streamlit_cmd = [str(venv_streamlit)]
            else:
                streamlit_cmd = ["streamlit"]
        else:
            streamlit_cmd = ["streamlit"]
        
        streamlit_cmd.extend(["run", str(streamlit_app_path)])

        # Start the Streamlit server
        try:
            print("⏳ Starting Streamlit server...")
            if platform.system() == "Windows":
                # On Windows, start in current console
                subprocess.run(streamlit_cmd)
            else:
                # On Unix/Mac, run directly
                subprocess.run(streamlit_cmd)
            
            return True
            
        except KeyboardInterrupt:
            print("\n🛑 Streamlit server stopped by user")
            return True
        except Exception as e:
            logger.error(f"Error starting Streamlit app: {e}")
            print(f"❌ Error starting Streamlit app: {e}")
            print("💡 Make sure Streamlit is installed: pip install streamlit")
            return False

    except Exception as e:
        logger.error(f"Unexpected error in run_streamlit_app: {e}")
        print(f"❌ Unexpected error: {e}")
        return False

def run_flask_app():
    """Run the Flask backend for AutoInsights Dashboard"""
    try:
        print("🚀 Starting AutoInsights Flask Backend...")
        print("📊 Loading Backend API")
        print(f"🌐 Backend will be available at: {Config.DASHBOARD_URL}")
        print("=" * 60)
        
        # Get the path to the Flask app
        flask_app_path = Config.FRONTEND_DIR / "backend_api.py"
        
        if not flask_app_path.exists():
            logger.error(f"Flask backend not found at {flask_app_path}")
            print("💡 Make sure the Frontend directory and backend_api.py exist")
            return False

        # Prepare the Flask command
        flask_cmd = [sys.executable, str(flask_app_path)]

        # Start the Flask server
        try:
            if platform.system() == "Windows":
                # On Windows, start in a new console window
                subprocess.Popen(
                    ['start', 'cmd', '/k'] + flask_cmd,
                    shell=True
                )
            else:
                # On Unix/Mac, run in background
                subprocess.Popen(flask_cmd)
            
            # Wait for server to start
            time.sleep(3)
            
            # Open browser
            webbrowser.open(Config.DASHBOARD_URL)
            
            logger.info("Flask server started successfully")
            print("✅ Flask server started successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error starting Flask app: {e}")
            print(f"❌ Error starting Flask app: {e}")
            print("💡 Make sure Flask is installed: pip install flask flask-cors")
            return False

    except Exception as e:
        logger.error(f"Unexpected error in run_flask_app: {e}")
        return False

def setup_project():
    """Set up the project environment"""
    print("�� Setting up AutoInsights Project...")
    
    # Ensure required directories exist
    directories = ["data", "output", "backups"]
    for dir_name in directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(exist_ok=True)
            print(f"📁 Created directory: {dir_name}")
    
    # Check for required input files only
    missing_files, empty_files = check_input_data_files()
    
    if missing_files or empty_files:
        print("⚠️  Missing or empty input data files detected:")
        for file in missing_files + empty_files:
            print(f"   - {file}")
        
        print("\n🔄 Generating sample data...")
        if generate_sample_data():
            print("✅ Sample data generated successfully!")
        else:
            print("❌ Failed to generate sample data")
            return False
    else:
        print("✅ All required input data files are present")
    
    return True

def main():
    """Main function to run AutoInsights Dashboard"""
    parser = argparse.ArgumentParser(description="AutoInsights Dashboard Launcher")
    parser.add_argument("--mode", choices=["streamlit", "flask", "setup"], 
                       default="streamlit", help="Choose the mode to run")
    parser.add_argument("--skip-setup", action="store_true", 
                       help="Skip initial setup and data generation")
    
    args = parser.parse_args()
    
    # Print welcome message
    print("=" * 60)
    print("🎯 AutoInsights Dashboard - Business Intelligence Platform")
    print("=" * 60)
    
    # Setup project if not skipped
    if not args.skip_setup:
        if not setup_project():
            print("❌ Project setup failed!")
            sys.exit(1)
    
    # Run the appropriate mode
    if args.mode == "streamlit":
        success = run_streamlit_app()
    elif args.mode == "flask":
        success = run_flask_app()
    elif args.mode == "setup":
        success = setup_project()
        if success:
            print("✅ Setup completed successfully!")
            print("💡 Run 'python run.py --mode streamlit' to start the dashboard")
        return
    else:
        print("❌ Invalid mode specified")
        sys.exit(1)
    
    if not success:
        print("❌ Failed to start AutoInsights Dashboard")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 AutoInsights Dashboard stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
