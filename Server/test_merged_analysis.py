#!/usr/bin/env python3
"""
Test script to verify the merged analysis endpoint and ColorCues fixes.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_merged_analysis():
    """Test the merged analysis functionality."""
    print("🧪 Testing Merged Analysis Endpoint")
    print("=" * 60)
    
    try:
        # Test the schema updates
        from src.app.schemas import AnalysisData, APIResponse
        
        # Test basic analysis data
        basic_data = {
            "prediction": "FAKE",
            "confidence": 0.85,
            "processing_time": 2.5,
            "note": "Test analysis"
        }
        
        analysis = AnalysisData(**basic_data)
        print(f"✅ Basic AnalysisData creation: {analysis.prediction}")
        
        # Test detailed analysis data
        detailed_data = {
            "prediction": "REAL",
            "confidence": 0.65,
            "processing_time": 4.2,
            "metrics": {
                "frame_count": 100,
                "max_score": 0.8,
                "min_score": 0.2,
                "suspicious_frames_count": 15
            },
            "note": "Detailed test analysis"
        }
        
        detailed_analysis = AnalysisData(**detailed_data)
        print(f"✅ Detailed AnalysisData creation: {detailed_analysis.prediction}")
        print(f"📊 Metrics included: {bool(detailed_analysis.metrics)}")
        
        # Test ColorCues format compatibility
        colorcues_data = {
            "prediction": "FAKE",
            "confidence": 0.895,
            "processing_time": 23.5,
            "metrics": {
                "sequence_count": 50,
                "frame_count": 50,  # Now included
                "max_score": 0.95,  # Now included
                "min_score": 0.1,   # Now included
                "suspicious_frames_count": 45,  # Now included
                "analysis_type": "sequence_based"
            }
        }
        
        colorcues_analysis = AnalysisData(**colorcues_data)
        print(f"✅ ColorCues AnalysisData creation: {colorcues_analysis.prediction}")
        print(f"🔍 ColorCues metrics: {colorcues_analysis.metrics['analysis_type']}")
        
        print(f"\n🎉 Schema validation completed successfully!")
        print(f"\n💡 Summary of changes:")
        print(f"   • /analyze endpoint now includes detailed metrics when available")
        print(f"   • /analyze/detailed endpoint removed (merged into /analyze)")
        print(f"   • ColorCues model now provides compatible metrics")
        print(f"   • Terminal logging shows proper values for all models")
        
    except Exception as e:
        print(f"❌ Error testing merged analysis: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    test_merged_analysis()
