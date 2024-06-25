import os
import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from ultralytics import YOLO
from scipy.stats import zscore


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def smooth_data(data, window_size=5, polyorder=2):
    if len(data) < window_size:
        return data  # No suficiente data para suavizar
    return savgol_filter(data, window_size, polyorder, axis=0)

def remove_outliers(data, z_thresh=3):
    z_scores = np.abs(zscore(data, axis=0, nan_policy='omit'))
    return data[(z_scores < z_thresh).all(axis=1)]


def extract_pose_features(video_path,model):
    cap = cv2.VideoCapture(video_path)
    all_features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        alto_original, ancho_original = frame.shape[:2]
        nuevo_alto = 700
        nuevo_ancho = int((nuevo_alto / alto_original) * ancho_original)
        frame = cv2.resize(frame, (nuevo_ancho, nuevo_alto))

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model(image)
        if not results or not results[0].keypoints:
            continue


        keypoints = results[0].keypoints.data.numpy().tolist()[0]

        if len(keypoints) == 17:
            try:
                left_shoulder = keypoints[5]
                right_shoulder = keypoints[6]
                left_elbow = keypoints[7]
                right_elbow = keypoints[8]
                left_wrist = keypoints[9]
                right_wrist = keypoints[10]
                left_hip = keypoints[11]
                right_hip = keypoints[12]
                left_knee = keypoints[13]
                right_knee = keypoints[14]
                left_ankle = keypoints[15]
                right_ankle = keypoints[16]


                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
                right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                
                wrist_distance = calculate_distance(left_wrist, right_wrist)
                ankle_distance = calculate_distance(left_ankle, right_ankle)
                shoulder_distance = calculate_distance(left_shoulder, right_shoulder)
                hip_distance = calculate_distance(left_hip, right_hip)
                
                features = np.array(keypoints).flatten().tolist()
                features += [left_elbow_angle, right_elbow_angle, left_shoulder_angle, right_shoulder_angle, left_knee_angle, right_knee_angle]
                features += [wrist_distance, ankle_distance, shoulder_distance, hip_distance]
                all_features.append(features)
            except Exception as e:
                print(f"Error procesando keypoints en video: {video_path} - {e}")

    cap.release()

    all_features = np.array(all_features)

    # if len(all_features) > 0:  # Verifica que haya datos para procesar
    #     all_features = smooth_data(all_features)  # Suavizar datos
    #     all_features = remove_outliers(all_features)  # Filtrar outliers

    return all_features

def process_videos(data_dir,model):
    all_data = []
    all_labels = []

    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        if os.path.isdir(category_path):
            for video in os.listdir(category_path):
                video_path = os.path.join(category_path, video)
                if video_path.endswith('.mp4'):
                    print(f"Procesando video: {video_path}")
                    data = extract_pose_features(video_path,model)
                    
                    if len(data) > 0:  # Verifica que haya datos para agregar
                        all_data.extend(data)
                        all_labels.extend([category] * len(data))

    return all_data, all_labels

def save_to_csv(data, labels, output_file):
    num_keypoints = 17 * 3
    columns = [f'keypoint_{i}' for i in range(num_keypoints)]
    columns += ['left_elbow_angle', 'right_elbow_angle', 'left_shoulder_angle', 'right_shoulder_angle', 'left_knee_angle', 'right_knee_angle']
    columns += ['wrist_distance', 'ankle_distance', 'shoulder_distance', 'hip_distance']
    
    df = pd.DataFrame(data, columns=columns)
    df['label'] = labels
    df.to_csv(output_file, index=False)

# Definir rutas
ruta_actual = os.getcwd()
carpeta_Yolo = os.path.dirname(ruta_actual)
ruta_raiz = os.path.dirname(carpeta_Yolo)

ruta_videos = os.path.join(ruta_raiz, 'Data Set', 'videos')
ruta_csv = os.path.join(carpeta_Yolo, 'Datos_Videos_Pose', 'pose_data.csv')

model = YOLO("yolov8n-pose.pt")

# Procesar videos y guardar a CSV
video_features, video_labels = process_videos(ruta_videos,model)
save_to_csv(video_features, video_labels, ruta_csv)
