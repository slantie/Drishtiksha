import requests

API_URL = "http://localhost:8000"
API_KEY = "98e75012eea105151f4ddbab76872dd3717c74247903b302f74abdd783d0c5bb"
VIDEO_PATH = "assets/id0_0001.mp4"


def test_analyze():
    print("Testing /analyze ...")
    with open(VIDEO_PATH, "rb") as f:
        files = {"video": (VIDEO_PATH, f, "video/mp4")}
        data = {"video_id": "test-analyze"}
        headers = {"X-API-Key": API_KEY}
        resp = requests.post(
            f"{API_URL}/analyze", files=files, data=data, headers=headers
        )
    print("Status:", resp.status_code)
    print("Response:", resp.json())


def test_analyze_visualize():
    print("Testing /analyze/visualize ...")
    with open(VIDEO_PATH, "rb") as f:
        files = {"video": (VIDEO_PATH, f, "video/mp4")}
        headers = {"X-API-Key": API_KEY}
        resp = requests.post(
            f"{API_URL}/analyze/visualize", files=files, headers=headers, stream=True
        )
    print("Status:", resp.status_code)
    if resp.status_code == 200:
        with open("visualized_output.mp4", "wb") as out:
            for chunk in resp.iter_content(chunk_size=8192):
                out.write(chunk)
        print("Visualized video saved as visualized_output.mp4")
    else:
        print("Response:", resp.text)


def test_analyze_detailed():
    print("Testing /analyze/detailed ...")
    with open(VIDEO_PATH, "rb") as f:
        files = {"video": (VIDEO_PATH, f, "video/mp4")}
        data = {"video_id": "test-detailed"}
        headers = {"X-API-Key": API_KEY}
        resp = requests.post(
            f"{API_URL}/analyze/detailed", files=files, data=data, headers=headers
        )
    print("Status:", resp.status_code)
    print("Response:", resp.json())


def test_analyze_frames():
    print("Testing /analyze/frames ...")
    with open(VIDEO_PATH, "rb") as f:
        files = {"video": (VIDEO_PATH, f, "video/mp4")}
        data = {"video_id": "test-frames"}
        headers = {"X-API-Key": API_KEY}
        resp = requests.post(
            f"{API_URL}/analyze/frames", files=files, data=data, headers=headers
        )
    print("Status:", resp.status_code)
    print("Response:", resp.json())


if __name__ == "__main__":
    test_analyze()
    print("-" * 40)
    test_analyze_visualize()
    print("-" * 40)
    test_analyze_detailed()
    print("-" * 40)
    test_analyze_frames()
