import numpy as np

def compute_angle(a, b, c):
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    )
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def compute_inclination(shoulder_mid, hip_mid):
    """
    Berechnet den Winkel des Torsos zur Vertikalen.
    0 Grad = Stehen (Kopf oben, Hüfte unten).
    90 Grad = Liegen (Push Up / Plank).
    """
    spine_vector = shoulder_mid - hip_mid
    vertical_vector = np.array([0, -1]) # Vektor zeigt nach oben (in Bild-Koord. ist y-negativ oben)
    
    spine_norm = np.linalg.norm(spine_vector)
    if spine_norm < 1e-6: 
        return 0.0
    
    cos_angle = np.dot(spine_vector, vertical_vector) / spine_norm
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))