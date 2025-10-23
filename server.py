"""Flask server for emotion detection web app."""
from typing import Dict, Optional
from flask import Flask, request, render_template
from EmotionDetection import emotion_detector

APP = Flask(__name__)

def _format_response_for_display(scores: Dict[str, Optional[float]]) -> str:
    """Format the emotion detection results for display."""
    if scores.get("dominant_emotion") is None:
        return "Invalid text! Please try again!"
    return (
        "For the given statement, the system response is "
        f"'anger': {scores['anger']}, 'disgust': {scores['disgust']}, "
        f"'fear': {scores['fear']}, 'joy': {scores['joy']} and "
        f"'sadness': {scores['sadness']}. The dominant emotion is "
        f"{scores['dominant_emotion']}."
    )

@APP.route("/", methods=["GET"])
def index():
    """Render the main HTML page."""
    return render_template("index.html")

@APP.route("/emotionDetector", methods=["GET"])
def emotion_detector_route():
    """Handle emotion detection requests."""
    user_text = request.args.get("textToAnalyze", "")
    scores = emotion_detector(user_text)
    if scores.get("dominant_emotion") is None:
        return "Invalid text! Please try again!", 200, {"Content-Type": "text/plain; charset=utf-8"}
    msg = _format_response_for_display(scores)
    return msg, 200, {"Content-Type": "text/plain; charset=utf-8"}

if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=5000, debug=False)
