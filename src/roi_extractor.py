def extract_roi(frame, landmarks, indices):
    h, w, _ = frame.shape
    xs, ys = [], []

    for idx in indices:
        xs.append(int(landmarks[idx].x * w))
        ys.append(int(landmarks[idx].y * h))

    return frame[min(ys):max(ys), min(xs):max(xs)]
