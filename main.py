#!/usr/bin/env python3
"""
Building Segmentation Pipeline - Main User Application

This is the main entry point for all user applications in the system.
"""

import os
import sys
import argparse
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / "scripts"))

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Building Segmentation Pipeline - Main Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Applications:
  pseudo-labels     Generate pseudo-labels from unlabeled images
  train            Train deep learning model with pseudo-labels
  inference        Run inference on new images
  api              Start API server
  frontend         Start web frontend
  docker           Manage Docker environment

Examples:
  python main.py pseudo-labels --help
  python main.py train --help
  python main.py inference --help
  python main.py docker start
        """
    )
    
    parser.add_argument(
        "application",
        choices=["pseudo-labels", "train", "inference", "api", "frontend", "docker"],
        help="Application to run"
    )
    
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to the selected application"
    )
    
    args = parser.parse_args()
    
    # Route to appropriate application
    if args.application == "pseudo-labels":
        from user_apps.create_pseudo_labels import main as pseudo_labels_main
        sys.argv = ["create_pseudo_labels.py"] + args.args
        pseudo_labels_main()
        
    elif args.application == "train":
        from user_apps.train_with_pseudo_labels import main as train_main
        sys.argv = ["train_with_pseudo_labels.py"] + args.args
        train_main()
        
    elif args.application == "inference":
        from inference.inference_engine import main as inference_main
        sys.argv = ["inference_engine.py"] + args.args
        inference_main()
        
    elif args.application == "api":
        from api.app import main as api_main
        sys.argv = ["app.py"] + args.args
        api_main()
        
    elif args.application == "frontend":
        from frontend.app import main as frontend_main
        sys.argv = ["app.py"] + args.args
        frontend_main()
        
    elif args.application == "docker":
        print("Docker management commands:")
        print("  ./docker-manage.sh start          # Start development environment")
        print("  ./docker-manage.sh start-prod     # Start production environment")
        print("  ./docker-manage.sh stop           # Stop services")
        print("  ./docker-manage.sh logs           # View logs")
        print("  ./docker-manage.sh status         # Check status")
        print("  ./docker-manage.sh scale api 3    # Scale API service")
        print("  ./docker-manage.sh cleanup        # Clean up environment")
        print("\nFor more options: ./docker-manage.sh help")

if __name__ == "__main__":
    main()
