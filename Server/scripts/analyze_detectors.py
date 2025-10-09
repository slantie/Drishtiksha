#!/usr/bin/env python3
"""
Script to update all detector files to support the generate_visualizations flag.

This script adds the generate_visualizations parameter to all detector analyze methods
and ensures visualization generation only happens when explicitly requested.
"""

import re
from pathlib import Path
from typing import List, Tuple

# Detectors that generate video visualizations (need updating)
VIDEO_DETECTORS = [
    "color_cues_detector.py",
    "efficientnet_detector.py",
    "eyeblink_detector.py",
    "cross_efficient_vit_detector.py",
    "distildire_detector.py",
    "lip_fd_detector.py",
    "mff_moe_detector.py",
]

# Audio detectors keep their spectrograms (different use case)
AUDIO_DETECTORS = [
    "mel_spectrogram_detector.py",
    "stft_spectrogram_detector.py",
    "scattering_wave_detector.py",
]

def find_visualization_calls(file_path: Path) -> List[Tuple[int, str]]:
    """Find all visualization generation calls in a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    visualization_calls = []
    for i, line in enumerate(lines, 1):
        if ('_generate_visualization' in line or 
            'VideoWriter' in line or
            'visualization_path =' in line):
            visualization_calls.append((i, line.strip()))
    
    return visualization_calls


def check_analyze_signature(file_path: Path) -> bool:
    """Check if analyze method has generate_visualizations parameter."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for analyze method signature
    pattern = r'def analyze\(self.*?\):'
    matches = re.findall(pattern, content, re.DOTALL)
    
    if not matches:
        return False
    
    for match in matches:
        if 'generate_visualizations' in match:
            return True
    
    return False


def analyze_detector_file(file_path: Path) -> dict:
    """Analyze a detector file and return its status."""
    result = {
        'file': file_path.name,
        'exists': file_path.exists(),
        'has_visualizations': False,
        'has_flag': False,
        'visualization_calls': [],
        'needs_update': False
    }
    
    if not file_path.exists():
        return result
    
    result['visualization_calls'] = find_visualization_calls(file_path)
    result['has_visualizations'] = len(result['visualization_calls']) > 0
    result['has_flag'] = check_analyze_signature(file_path)
    
    # Needs update if it has visualizations but no flag
    result['needs_update'] = result['has_visualizations'] and not result['has_flag']
    
    return result


def main():
    """Main analysis function."""
    print("=" * 80)
    print("DETECTOR VISUALIZATION FLAG ANALYSIS")
    print("=" * 80)
    print()
    
    detectors_dir = Path(__file__).parent.parent / "src" / "ml" / "detectors"
    
    if not detectors_dir.exists():
        print(f"ERROR: Detectors directory not found: {detectors_dir}")
        return
    
    print(f"Analyzing detectors in: {detectors_dir}")
    print()
    
    # Analyze all detectors
    print("📹 VIDEO DETECTORS (need updating):")
    print("-" * 80)
    
    video_results = []
    for detector_name in VIDEO_DETECTORS:
        detector_path = detectors_dir / detector_name
        result = analyze_detector_file(detector_path)
        video_results.append(result)
        
        status = "✅ UPDATED" if result['has_flag'] else ("⚠️  NEEDS UPDATE" if result['needs_update'] else "✓ OK")
        print(f"{status:15} {detector_name:40} Visualizations: {len(result['visualization_calls'])}")
    
    print()
    print("🎵 AUDIO DETECTORS (keep spectrograms):")
    print("-" * 80)
    
    audio_results = []
    for detector_name in AUDIO_DETECTORS:
        detector_path = detectors_dir / detector_name
        result = analyze_detector_file(detector_path)
        audio_results.append(result)
        
        status = "✅ HAS FLAG" if result['has_flag'] else "ℹ️  NO FLAG"
        print(f"{status:15} {detector_name:40} Spectrograms: {len(result['visualization_calls'])}")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    video_needs_update = [r for r in video_results if r['needs_update']]
    video_updated = [r for r in video_results if r['has_flag']]
    
    print(f"Video Detectors:")
    print(f"  - Total: {len(video_results)}")
    print(f"  - Updated with flag: {len(video_updated)}")
    print(f"  - Need update: {len(video_needs_update)}")
    
    if video_needs_update:
        print()
        print("⚠️  FILES THAT NEED UPDATING:")
        for result in video_needs_update:
            print(f"   - {result['file']}")
            for line_num, line_content in result['visualization_calls'][:3]:
                print(f"      Line {line_num}: {line_content[:60]}...")
    
    print()
    print(f"Audio Detectors: {len(audio_results)} (spectrograms are kept)")
    
    print()
    print("=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("""
For each detector that needs updating:

1. Add 'generate_visualizations: bool = False' parameter to analyze() method:
   ```python
   def analyze(self, media_path: str, generate_visualizations: bool = False, **kwargs):
   ```

2. Wrap visualization generation in a conditional:
   ```python
   visualization_path = None
   if generate_visualizations:
       visualization_path = self._generate_visualization(...)
   else:
       logger.info(f"Skipping visualization generation")
   ```

3. Pass the flag through to helper methods if needed.

4. Test that the detector works with both True and False values.
""")


if __name__ == "__main__":
    main()
