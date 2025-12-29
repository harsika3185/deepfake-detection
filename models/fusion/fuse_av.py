def fuse_audio_visual(visual_score, audio_score,
                      w_visual=0.6, w_audio=0.4):
    """
    Simple weighted audio-visual fusion.

    Args:
        visual_score (float): visual fake probability
        audio_score (float): audio fake probability
        w_visual (float): weight for visual modality
        w_audio (float): weight for audio modality

    Returns:
        final_score (float)
        decision (str)
    """

    final_score = (w_visual * visual_score) + (w_audio * audio_score)
    decision = "FAKE" if final_score >= 0.5 else "REAL"

    return final_score, decision


if __name__ == "__main__":
    # Example values (replace with real outputs)
    visual_prob = 0.4375   # from run_video_visual.py
    audio_prob = 0.4907    # from test_audio.py

    score, decision = fuse_audio_visual(visual_prob, audio_prob)

    print("Visual probability:", visual_prob)
    print("Audio probability :", audio_prob)
    print("Final fused score :", round(score, 4))
    print("FINAL DECISION    :", decision)
